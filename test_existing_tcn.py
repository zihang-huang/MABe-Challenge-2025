import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Configuration
DATA_DIR = './data/processed/train'
BATCH_SIZE = 1 # Batch size 1 for validation/visualization
SEQ_LEN = 100
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2
MODEL_PATH = 'model_final.pth' # Using the final trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Classes (Same as before) ---

class MouseBehaviorDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab=None, seq_len=100, is_training=True):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.seq_len = seq_len
        self.is_training = is_training
        
        self.feature_cols = self._get_feature_columns()
        if behavior_vocab is None:
            self.behavior_vocab = self._build_vocab()
        else:
            self.behavior_vocab = behavior_vocab
            
        self.num_classes = len(self.behavior_vocab)
        self.num_features = len(self.feature_cols)
        self.scaler = self._get_scaler()

    def _get_feature_columns(self):
        if not self.video_ids:
            return []
        sample_vid = self.video_ids[0]
        try:
            df = pd.read_parquet(os.path.join(self.data_dir, f"{sample_vid}.parquet"))
        except Exception as e:
            print(f"Error reading sample video {sample_vid}: {e}")
            return []
            
        exclude = ['video_frame', 'mouse_id', 'video_id', 'lab_id']
        cols = [c for c in df.columns if c not in exclude]
        base_cols = [c for c in cols if not c.startswith('dist_to_mouse')] 
        for i in range(1, 5):
            base_cols.append(f'dist_to_mouse_{i}')
        return base_cols

    def _build_vocab(self):
        behaviors = set()
        scan_ids = [vid for vid in self.video_ids if os.path.exists(os.path.join(self.data_dir, f"{vid}_annotations.parquet"))]
        scan_ids = scan_ids[:100] 
        for vid in scan_ids:
            anno_path = os.path.join(self.data_dir, f"{vid}_annotations.parquet")
            if os.path.exists(anno_path):
                df = pd.read_parquet(anno_path)
                behaviors.update(df['action'].unique())
        return sorted(list(behaviors))

    def _get_scaler(self):
        scaler_path = os.path.join(self.data_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        return None

class MouseIndexDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab, seq_len=100, is_training=True):
        self.base_dataset = MouseBehaviorDataset(data_dir, video_ids, behavior_vocab, seq_len, is_training)
        self.indices = []
        
        for i, vid in enumerate(video_ids):
            meta_path = os.path.join(data_dir, f"{vid}_meta.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                num_mice = meta.get('num_mice', 2)
                for m in range(num_mice):
                    self.indices.append((i, m + 1))
            except:
                pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        vid_idx, mouse_id = self.indices[idx]
        vid = self.base_dataset.video_ids[vid_idx]
        
        feat_path = os.path.join(self.base_dataset.data_dir, f"{vid}.parquet")
        feat_df = pd.read_parquet(feat_path)
        
        for col in self.base_dataset.feature_cols:
            if col not in feat_df.columns: feat_df[col] = 0.0
                
        max_frame = feat_df['video_frame'].max()
        m_df = feat_df[feat_df['mouse_id'] == mouse_id].sort_values('video_frame')
        m_df = m_df.set_index('video_frame').reindex(range(int(max_frame) + 1), fill_value=0).reset_index()
        
        feats = m_df[self.base_dataset.feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats)
        
        if self.base_dataset.scaler:
            feats = self.base_dataset.scaler.transform(feats)
            
        labels = np.zeros((len(feats), self.base_dataset.num_classes), dtype=np.float32)
        
        # Load Real Annotations
        anno_path = os.path.join(self.base_dataset.data_dir, f"{vid}_annotations.parquet")
        if os.path.exists(anno_path):
            self._fill_labels(labels, anno_path, mouse_id)

        # Windowing logic matching training/inference
        if self.base_dataset.is_training and len(feats) > self.base_dataset.seq_len:
            start_idx = np.random.randint(0, len(feats) - self.base_dataset.seq_len)
            feats = feats[start_idx : start_idx + self.base_dataset.seq_len]
            labels = labels[start_idx : start_idx + self.base_dataset.seq_len]
        elif len(feats) < self.base_dataset.seq_len:
            pad_len = self.base_dataset.seq_len - len(feats)
            feats = np.pad(feats, ((0, pad_len), (0, 0)))
            labels = np.pad(labels, ((0, pad_len), (0, 0)))
            
        return torch.tensor(feats.T), torch.tensor(labels), vid, mouse_id, torch.tensor(max_frame)

    def _fill_labels(self, labels, path, mouse_id):
        try:
            df = pd.read_parquet(path)
            mouse_annos = df[df['agent_id'] == mouse_id]
            for _, row in mouse_annos.iterrows():
                action = row['action']
                if action in self.base_dataset.behavior_vocab:
                    act_idx = self.base_dataset.behavior_vocab.index(action)
                    start = max(0, int(row['start_frame']))
                    stop = min(len(labels) - 1, int(row['stop_frame']))
                    labels[start:stop+1, act_idx] = 1.0
        except Exception as e:
            print(f"Error reading labels from {path}: {e}")

# --- Model (Same as before) ---

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[2] != res.shape[2]: out = out[:, :, :res.shape[2]]
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, num_classes=10):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(num_channels[-1], num_classes, 1)

    def forward(self, x):
        y = self.network(x)
        y = self.classifier(y)
        return y

# --- Evaluation Functions ---

def evaluate_model(model, val_loader, vocab):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nEvaluating model on validation set...")
    with torch.no_grad():
        for features, labels, _, _, _ in tqdm(val_loader, desc="Evaluation"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features) # (B, C, T)
            probs = torch.sigmoid(outputs) # (B, C, T)
            
            # Flatten to (Total_Time, Classes)
            probs_flat = probs.permute(0, 2, 1).reshape(-1, len(vocab)).cpu().numpy()
            labels_flat = labels.reshape(-1, len(vocab)).cpu().numpy()
            
            # Simple thresholding at 0.5
            preds_bin = (probs_flat > 0.5).astype(int)
            
            all_preds.append(preds_bin)
            all_targets.append(labels_flat)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=vocab, zero_division=0))

def visualize_sample(model, dataset, vocab, save_path="visualization.png"):
    model.eval()
    # Pick a random sample with some activity
    found = False
    max_attempts = 20
    
    print("\nLooking for an interesting sample to visualize...")
    
    for _ in range(max_attempts):
        idx = np.random.randint(0, len(dataset))
        features, labels, vid, mouse_id, max_frame = dataset[idx]
        
        # Check if there are any positive labels in this window
        if labels.sum() > 0:
            found = True
            break
            
    if not found:
        print("Could not find a sample with active behaviors in random search. Using last sampled index.")
    
    features_in = features.unsqueeze(0).to(device) # (1, C, T)
    with torch.no_grad():
        output = model(features_in)
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy() # (C, T)
        
        labels = labels.cpu().numpy().T # (C, T)
    
    # Plot
    fig, axes = plt.subplots(len(vocab), 1, figsize=(12, 2 * len(vocab)), sharex=True)
    if len(vocab) == 1: axes = [axes]
    
    time_steps = np.arange(probs.shape[1])
    
    for i, ax in enumerate(axes):
        ax.plot(time_steps, labels[i], label='Ground Truth', color='green', alpha=0.6, linewidth=2)
        ax.plot(time_steps, probs[i], label='Prediction', color='red', linestyle='--', alpha=0.8)
        ax.set_ylabel(vocab[i])
        ax.set_ylim(-0.1, 1.1)
        if i == 0:
            ax.legend(loc='upper right')
            ax.set_title(f"Video: {vid}, Mouse: {mouse_id}")
            
    plt.xlabel("Frame (Window)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

# --- Main ---

def get_video_lists(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    video_ids = [os.path.basename(f).replace(".parquet", "") for f in all_files if not f.endswith("_annotations.parquet")]
    
    labeled = []
    
    for vid in video_ids:
        if os.path.exists(os.path.join(data_dir, f"{vid}_annotations.parquet")):
            labeled.append(vid)
            
    return labeled

def main():
    labeled_vids = get_video_lists(DATA_DIR)
    print(f"Found {len(labeled_vids)} labeled videos.")
    
    # Split labeled for validation
    train_vids, val_vids = train_test_split(labeled_vids, test_size=0.1, random_state=42)
    print(f"Using {len(val_vids)} videos for validation/testing.")
    
    # Initialize Dataset to get Vocab and Params
    base_ds = MouseBehaviorDataset(DATA_DIR, labeled_vids)
    vocab = base_ds.behavior_vocab
    print(f"Vocabulary ({len(vocab)}): {vocab}")
    
    val_ds = MouseIndexDataset(DATA_DIR, val_vids, vocab, seq_len=SEQ_LEN, is_training=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialize Model
    channel_sizes = [HIDDEN_CHANNELS] * 4
    model = TCN(num_inputs=base_ds.num_features, num_channels=channel_sizes, 
                kernel_size=KERNEL_SIZE, dropout=DROPOUT, num_classes=len(vocab)).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Error: Model file {MODEL_PATH} not found!")
        return
        
    # Evaluate
    evaluate_model(model, val_loader, vocab)
    
    # Visualize
    visualize_sample(model, val_ds, vocab, save_path="final_prediction_visualization.png")

if __name__ == "__main__":
    main()
