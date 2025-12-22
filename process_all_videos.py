import os
import pandas as pd

TRACK_ROOT = "data/train_tracking"
ANN_ROOT   = "data/train_annotation"
OUT_DIR    = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "frame" in df.columns:
        return df
    if df.index.name == "frame":
        return df.reset_index()
    df = df.reset_index(drop=True)
    df["frame"] = df.index
    return df


processed = 0
skipped = 0

for dataset in sorted(os.listdir(ANN_ROOT)):
    ann_dataset_dir = os.path.join(ANN_ROOT, dataset)
    track_dataset_dir = os.path.join(TRACK_ROOT, dataset)

    if not os.path.isdir(ann_dataset_dir):
        continue
    if not os.path.isdir(track_dataset_dir):
        print(f"Skipping dataset {dataset}: no tracking dir")
        continue

    # iterate over VIDEO files
    for fname in os.listdir(ann_dataset_dir):
        if not fname.endswith(".parquet"):
            continue

        video_id = fname.replace(".parquet", "")

        ann_path = os.path.join(ann_dataset_dir, fname)
        track_path = os.path.join(track_dataset_dir, f"{video_id}.parquet")

        if not os.path.exists(track_path):
            skipped += 1
            continue

        try:
            ann = pd.read_parquet(ann_path)
            track = pd.read_parquet(track_path)
        except Exception as e:
            print(f"Skipping {dataset}/{video_id}: read error {e}")
            skipped += 1
            continue

        track = normalize_frame(track)

        required_ann = {"agent_id", "target_id", "action", "start_frame", "stop_frame"}
        if not required_ann.issubset(ann.columns):
            print(f"Skipping {dataset}/{video_id}: bad annotation schema")
            skipped += 1
            continue

        track_out = os.path.join(OUT_DIR, f"preprocessed_{video_id}.parquet")
        ann_out   = os.path.join(OUT_DIR, f"annotations_{video_id}.parquet")

        track.to_parquet(track_out, index=False)
        ann.to_parquet(ann_out, index=False)

        processed += 1

print(f"✅ Processed videos: {processed}")
print(f"⚠️  Skipped videos: {skipped}")
