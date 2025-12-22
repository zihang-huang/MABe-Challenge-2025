# ============================================================
# tcn_behavior_model_final.py
# Train TCN on data/processed/train_frames.csv and create
# submission.csv in the required segment format.
# ============================================================

import os
import gc
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


# ----------------------------
# Config
# ----------------------------

@dataclass
class CFG:
    # paths
    train_frames_csv: str = "data/processed/train_frames.csv"
    test_meta_csv: str = "data/test.csv"  # metadata only; used to get test video_ids
    proc_dir: str = "data/processed"      # contains preprocessed_<video_id>.parquet for test
    sample_submission_csv: str = "data/sample_submission.csv"
    out_dir: str = "outputs"
    submission_out: str = "outputs/submission.csv"
    label_map_out: str = "outputs/label_map.json"

    # columns
    id_cols: Tuple[str, ...] = ("video_id", "agent_id", "target_id", "frame")
    target_col: str = "action"

    # windowing
    window: int = 64
    stride: int = 16

    # training
    n_folds: int = 5
    epochs: int = 8
    batch_size: int = 128
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    seed: int = 42

    # model
    hidden: int = 128
    dropout: float = 0.1

    # inference → segments
    min_segment_len: int = 2  # frames
    background_labels: Tuple[str, ...] = ("none", "no_action", "background", "other")


CFG = CFG()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Reproducibility
# ----------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Window builder
# ----------------------------

def build_windows(df: pd.DataFrame, group_cols: List[str], window: int, stride: int) -> List[Tuple[int, int]]:
    df = df.sort_values(group_cols + ["frame"]).reset_index(drop=True)
    windows: List[Tuple[int, int]] = []
    for _, g in df.groupby(group_cols, sort=False):
        idx = g.index.to_numpy()
        if len(idx) < window:
            continue
        for i in range(0, len(idx) - window + 1, stride):
            windows.append((int(idx[i]), int(idx[i + window - 1])))
    return windows


# ----------------------------
# Dataset
# ----------------------------

class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        windows: List[Tuple[int, int]],
        feature_cols: List[str],
        target_col: Optional[str] = None,
    ):
        self.df = df
        self.windows = windows
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        s, e = self.windows[i]
        x = self.df.loc[s:e, self.feature_cols].to_numpy(np.float32)  # (T,F)
        if self.target_col is None:
            return torch.from_numpy(x)
        y = int(self.df.loc[(s + e) // 2, self.target_col])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ----------------------------
# Model (TCN)
# ----------------------------

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B,C,T)
        y = self.conv1(x)
        y = y[..., : x.shape[-1]]
        y = self.act(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = y[..., : x.shape[-1]]
        y = self.act(y)
        y = self.drop(y)

        return y + self.res(x)


class TCNClassifier(nn.Module):
    def __init__(self, n_feats: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(n_feats, hidden, dilation=1, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=4, dropout=dropout),
            TCNBlock(hidden, hidden, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        x = x.permute(0, 2, 1)
        h = self.net(x)  # (B,H,T)
        h = h[:, :, h.shape[-1] // 2]  # center pooling
        return self.head(h)


# ----------------------------
# Train / eval
# ----------------------------

def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


# ----------------------------
# IO helpers
# ----------------------------

def ensure_frame_column(df: pd.DataFrame) -> pd.DataFrame:
    if "frame" in df.columns:
        return df
    if df.index.name == "frame":
        return df.reset_index()
    df = df.reset_index(drop=True)
    df["frame"] = df.index
    return df


def load_test_frames_for_video(proc_dir: str, video_id: str) -> pd.DataFrame:
    # expects preprocessed_<video_id>.parquet
    path = os.path.join(proc_dir, f"preprocessed_{video_id}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df = ensure_frame_column(df)

    # must have agent_id/target_id to match training rows
    if "agent_id" not in df.columns or "target_id" not in df.columns:
        raise ValueError(
            f"{path} must contain agent_id and target_id columns (pair-level features). "
            f"Found columns: {list(df.columns)[:30]}"
        )

    df["video_id"] = str(video_id)
    return df


def background_class_ids(le: LabelEncoder, background_labels: Tuple[str, ...]) -> set:
    s = set()
    classes = list(le.classes_)
    for i, c in enumerate(classes):
        if str(c).lower() in set([b.lower() for b in background_labels]):
            s.add(i)
    return s


def frames_to_segments(
    df: pd.DataFrame,
    action_id_col: str,
    le: LabelEncoder,
    min_len: int,
    bg_ids: set,
) -> pd.DataFrame:
    # df includes: video_id, agent_id, target_id, frame, action_id
    rows = []
    for (vid, a, t), g in df.sort_values(["video_id", "agent_id", "target_id", "frame"]).groupby(
        ["video_id", "agent_id", "target_id"], sort=False
    ):
        frames = g["frame"].to_numpy()
        acts = g[action_id_col].to_numpy()

        if len(frames) == 0:
            continue

        start = int(frames[0])
        cur = int(acts[0])

        for i in range(1, len(frames)):
            if int(acts[i]) != cur:
                stop = int(frames[i - 1])
                if (stop - start + 1) >= min_len and cur not in bg_ids:
                    rows.append(
                        {
                            "video_id": int(vid) if str(vid).isdigit() else vid,
                            "agent_id": a,
                            "target_id": t,
                            "action": str(le.inverse_transform([cur])[0]),
                            "start_frame": start,
                            "stop_frame": stop,
                        }
                    )
                start = int(frames[i])
                cur = int(acts[i])

        # last segment
        stop = int(frames[-1])
        if (stop - start + 1) >= min_len and cur not in bg_ids:
            rows.append(
                {
                    "video_id": int(vid) if str(vid).isdigit() else vid,
                    "agent_id": a,
                    "target_id": t,
                    "action": str(le.inverse_transform([cur])[0]),
                    "start_frame": start,
                    "stop_frame": stop,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # sort segments and create row_id
    out = out.sort_values(["video_id", "agent_id", "target_id", "start_frame"]).reset_index(drop=True)
    out.insert(0, "row_id", np.arange(len(out), dtype=int))
    return out


# ----------------------------
# Main pipeline
# ----------------------------

def train_cv_and_save(cfg: CFG) -> Dict[str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    df = pd.read_csv(cfg.train_frames_csv)

    # enforce ids exist
    for c in cfg.id_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in train_frames: {c}")

    # label encode
    le = LabelEncoder()
    df["action_id"] = le.fit_transform(df[cfg.target_col].astype(str))
    n_classes = len(le.classes_)

    # save label map
    label_map = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
    with open(cfg.label_map_out, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # features: numeric only
    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # remove non-input numeric columns
    for c in ["frame", "action_id"]:
        if c in feature_cols:
            feature_cols.remove(c)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found")

    # hard safety checks
    assert "bodypart" not in feature_cols
    assert cfg.target_col not in feature_cols


    # window index
    group_cols = ["video_id", "agent_id", "target_id"]
    windows = build_windows(df, group_cols, cfg.window, cfg.stride)
    if len(windows) == 0:
        raise ValueError("No windows built. Check frame continuity and window/stride.")

    # group for CV by video
    window_video = df.loc[[s for (s, _) in windows], "video_id"].astype(str).to_numpy()

    n_groups = df["video_id"].nunique()
    n_splits = min(cfg.n_folds, n_groups)

    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 videos for GroupKFold, found {n_groups}. "
            "Preprocess more videos or reduce n_folds."
        )

    gkf = GroupKFold(n_splits=n_splits)


    fold_paths = {}

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(len(windows)), groups=window_video)):
        print(f"\nFOLD {fold}")

        tr_w = [windows[i] for i in tr_idx]
        va_w = [windows[i] for i in va_idx]

        ds_tr = WindowDataset(df, tr_w, feature_cols, target_col="action_id")
        ds_va = WindowDataset(df, va_w, feature_cols, target_col="action_id")

        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        model = TCNClassifier(len(feature_cols), n_classes, hidden=cfg.hidden, dropout=cfg.dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_state = None

        for ep in range(cfg.epochs):
            tr_loss = train_one_epoch(model, dl_tr, opt, loss_fn)
            # validation loss proxy (no metric dependency)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in dl_va:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    val_losses.append(float(loss.item()))
            va_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            print(f"epoch {ep+1:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f}")

            if va_loss < best_loss:
                best_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # save fold weights
        fold_path = os.path.join(cfg.out_dir, f"tcn_fold_{fold}.pt")
        torch.save(best_state, fold_path)
        fold_paths[str(fold)] = fold_path
        print("saved", fold_path)

        del model, ds_tr, ds_va, dl_tr, dl_va
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "label_map": cfg.label_map_out,
        "fold_paths": fold_paths,
        "train_frames": cfg.train_frames_csv,
    }


def infer_and_write_submission(cfg: CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # load label map and rebuild encoder in the same order
    with open(cfg.label_map_out, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # label_map keys are ints as strings
    classes = [label_map[str(i)] for i in range(len(label_map))]
    le = LabelEncoder()
    le.fit(classes)

    bg_ids = background_class_ids(le, cfg.background_labels)

    # discover folds
    fold_paths = []
    for fold in range(cfg.n_folds):
        p = os.path.join(cfg.out_dir, f"tcn_fold_{fold}.pt")
        if os.path.exists(p):
            fold_paths.append(p)
    if len(fold_paths) == 0:
        raise FileNotFoundError("No fold weights found in outputs/. Train first.")

    # get test videos from metadata
    test_meta = pd.read_csv(cfg.test_meta_csv)
    if "video_id" not in test_meta.columns:
        raise ValueError("data/test.csv must contain a video_id column")
    test_video_ids = test_meta["video_id"].astype(str).tolist()

    # build per-frame predictions for each test video, then segments
    all_frame_preds = []

    # rebuild feature_cols exactly as in training
    train_df = pd.read_csv(cfg.train_frames_csv, nrows=1000)
    feature_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
    for c in ["frame", "action_id"]:
        if c in feature_cols:
            feature_cols.remove(c)


    n_classes = len(le.classes_)

    for vid in test_video_ids:
        feat = load_test_frames_for_video(cfg.proc_dir, vid)

        # keep only required ids + features
        missing = [c for c in feature_cols if c not in feat.columns]
        if missing:
            raise ValueError(f"Missing feature columns in test preprocessed_{vid}.parquet: {missing[:15]}")

        # build windows on test
        group_cols = ["video_id", "agent_id", "target_id"]
        windows = build_windows(feat, group_cols, cfg.window, cfg.stride)
        if len(windows) == 0:
            continue

        ds = WindowDataset(feat, windows, feature_cols, target_col=None)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        # ensemble over folds
        probs_sum = None
        for pth in fold_paths:
            model = TCNClassifier(len(feature_cols), n_classes, hidden=cfg.hidden, dropout=cfg.dropout).to(DEVICE)
            state = torch.load(pth, map_location="cpu")
            model.load_state_dict(state, strict=True)
            pr = predict_proba(model, dl)
            probs_sum = pr if probs_sum is None else (probs_sum + pr)
            del model
            torch.cuda.empty_cache()

        probs = probs_sum / len(fold_paths)

        # assign each window prediction to its center frame row
        centers = np.array([(s + e) // 2 for (s, e) in windows], dtype=int)
        center_rows = feat.loc[centers, ["video_id", "agent_id", "target_id", "frame"]].copy()
        action_id = probs.argmax(axis=1).astype(int)
        center_rows["action_id"] = action_id

        all_frame_preds.append(center_rows)

    if len(all_frame_preds) == 0:
        # still write an empty, schema-correct submission
        sub = pd.read_csv(cfg.sample_submission_csv).iloc[0:0].copy()
        sub.to_csv(cfg.submission_out, index=False)
        print("No predictions generated; wrote empty submission:", cfg.submission_out)
        return

    frame_pred_df = pd.concat(all_frame_preds, ignore_index=True)

    # convert frame predictions → segments (matches sample_submission schema)
    seg_df = frames_to_segments(
        frame_pred_df,
        action_id_col="action_id",
        le=le,
        min_len=cfg.min_segment_len,
        bg_ids=bg_ids,
    )

    # align to exact column order
    if seg_df.empty:
        sub = pd.read_csv(cfg.sample_submission_csv).iloc[0:0].copy()
        sub.to_csv(cfg.submission_out, index=False)
        print("Only background predicted; wrote empty submission:", cfg.submission_out)
        return

    # Ensure same columns as sample
    sample = pd.read_csv(cfg.sample_submission_csv)
    out = seg_df.copy()

    # If sample has stricter dtypes, keep columns only
    out = out[sample.columns.tolist()]

    out.to_csv(cfg.submission_out, index=False)
    print("Wrote submission:", cfg.submission_out, "rows:", len(out))


def main():
    # Train if fold weights missing; always attempt submission after training
    os.makedirs(CFG.out_dir, exist_ok=True)
    need_train = any(not os.path.exists(os.path.join(CFG.out_dir, f"tcn_fold_{f}.pt")) for f in range(CFG.n_folds))
    if need_train:
        train_cv_and_save(CFG)
    infer_and_write_submission(CFG)


if __name__ == "__main__":
    main()

