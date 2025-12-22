import os
import pandas as pd

PROC_DIR = "data/processed"
OUT_PATH = "data/processed/train_frames.csv"

rows = []
used = 0

for fname in os.listdir(PROC_DIR):
    if not fname.startswith("annotations_") or not fname.endswith(".parquet"):
        continue

    video_id = fname.replace("annotations_", "").replace(".parquet", "")
    ann_path = os.path.join(PROC_DIR, fname)
    feat_path = os.path.join(PROC_DIR, f"preprocessed_{video_id}.parquet")

    if not os.path.exists(feat_path):
        continue

    ann = pd.read_parquet(ann_path)
    feat = pd.read_parquet(feat_path)

    # normalize frame
    if "frame" not in feat.columns:
        if feat.index.name == "frame":
            feat = feat.reset_index()
        else:
            feat = feat.reset_index(drop=True)
            feat["frame"] = feat.index

    required_ann = {"agent_id", "target_id", "action", "start_frame", "stop_frame"}
    if not required_ann.issubset(ann.columns):
        raise ValueError(f"Annotation schema mismatch in {ann_path}")

    for _, seg in ann.iterrows():
        mask = (feat["frame"] >= seg["start_frame"]) & (feat["frame"] <= seg["stop_frame"])
        frames = feat.loc[mask].copy()
        if frames.empty:
            continue

        frames["video_id"] = video_id
        frames["agent_id"] = seg["agent_id"]
        frames["target_id"] = seg["target_id"]
        frames["action"] = seg["action"]

        rows.append(frames)

    used += 1
    if used % 25 == 0:
        print("processed videos:", used)

if not rows:
    raise RuntimeError("No rows generated. Check data/processed contents.")

df = pd.concat(rows, ignore_index=True)
df.to_csv(OUT_PATH, index=False)

print("Wrote", OUT_PATH, "rows:", len(df), "videos:", used)
