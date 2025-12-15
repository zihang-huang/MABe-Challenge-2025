
import os
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Configuration
DATA_DIR = './data/processed/train'
BATCH_SIZE = 64
SEQ_LEN = 100
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2
LR = 0.001
EPOCHS = 10 # Sufficient for supervised check

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Classes ---

class MouseBehaviorDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab=None, seq_len=100, is_training=True, scaler=None):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.seq_len = seq_len
        self.is_training = is_training
        
        self.feature_cols = self._get_feature_columns()
        if behavior_vocab is None:
            self.behavior_vocab, self.class_counts = self._build_vocab()
        else:
            self.behavior_vocab = behavior_vocab
            self.class_counts = None
            
        self.num_classes = len(self.behavior_vocab)
        self.num_features = len(self.feature_cols)
        
        # Handle Scaler: If provided, use it. If training and None, fit it.
        self.scaler = scaler
        if self.is_training and self.scaler is None:
            self.scaler = self._fit_scaler()

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
        print("Building vocabulary from all videos...")
        counts = {}
        # Scan all videos, not just first 100
        for vid in tqdm(self.video_ids, desc="Scanning Vocab"):
            anno_path = os.path.join(self.data_dir, f"{vid}_annotations.parquet")
            if os.path.exists(anno_path):
                try:
                    df = pd.read_parquet(anno_path)
                    if 'action' in df.columns and 'start_frame' in df.columns and 'stop_frame' in df.columns:
                        durations = (df['stop_frame'] - df['start_frame'] + 1).values
                        actions = df['action'].values
                        for act, dur in zip(actions, durations):
                            counts[act] = counts.get(act, 0) + dur
                except Exception as e:
                    print(f"Error reading {anno_path}: {e}")
        
        vocab = sorted(list(counts.keys()))
        vocab_counts = [counts[v] for v in vocab]
        print(f"Found {len(vocab)} behaviors.")
        return vocab, vocab_counts

    def _fit_scaler(self):
        print("Fitting scaler on training data...")
        scaler = StandardScaler()
        # Use a subset of videos to fit scaler to save time
        sample_size = min(len(self.video_ids), 200)
        sample_vids = np.random.choice(self.video_ids, sample_size, replace=False)
        all_feats = []
        
        for vid in tqdm(sample_vids, desc="Scaler Data"):
            feat_path = os.path.join(self.data_dir, f"{vid}.parquet")
            if os.path.exists(feat_path):
                df = pd.read_parquet(feat_path)
                for col in self.feature_cols:
                    if col not in df.columns: df[col] = 0.0
                feats = df[self.feature_cols].values.astype(np.float32)
                feats = np.nan_to_num(feats)
                # Subsample frames
                all_feats.append(feats[::10])
        
        if all_feats:
            concat_feats = np.concatenate(all_feats, axis=0)
            scaler.fit(concat_feats)
            return scaler
        return None

class MouseIndexDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.indices = []
        
        for i, vid in enumerate(base_dataset.video_ids):
            meta_path = os.path.join(base_dataset.data_dir, f"{vid}_meta.json")
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
        
        # Load Annotations
        anno_path = os.path.join(self.base_dataset.data_dir, f"{vid}_annotations.parquet")
        if os.path.exists(anno_path):
            # removed try-except to debug
            df = pd.read_parquet(anno_path)
            
            # Check types
            # print(f"DEBUG: mouse_id={mouse_id} type={type(mouse_id)}, agent_id type={df['agent_id'].dtype}")
            
            # Force same type (int)
            mouse_id = int(mouse_id)
            if df['agent_id'].dtype == 'object':
                 df['agent_id'] = df['agent_id'].astype(int)
            
            mouse_annos = df[df['agent_id'] == mouse_id]
            for _, row in mouse_annos.iterrows():
                action = row['action']
                if action in self.base_dataset.behavior_vocab:
                    act_idx = self.base_dataset.behavior_vocab.index(action)
                    start = max(0, int(row['start_frame']))
                    stop = min(len(labels) - 1, int(row['stop_frame']))
                    labels[start:stop+1, act_idx] = 1.0

        # Windowing
        if self.base_dataset.is_training and len(feats) > self.base_dataset.seq_len:
            start_idx = np.random.randint(0, len(feats) - self.base_dataset.seq_len)
            feats = feats[start_idx : start_idx + self.base_dataset.seq_len]
            labels = labels[start_idx : start_idx + self.base_dataset.seq_len]
        elif len(feats) < self.base_dataset.seq_len:
            pad_len = self.base_dataset.seq_len - len(feats)
            feats = np.pad(feats, ((0, pad_len), (0, 0)))
            labels = np.pad(labels, ((0, pad_len), (0, 0)))
            
        return torch.tensor(feats.T), torch.tensor(labels), vid, mouse_id, torch.tensor(max_frame)

# --- Model ---

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

# --- Functions ---

def train_model(model, train_loader, val_loader, epochs, save_path, class_counts=None):
    num_classes = model.classifier.out_channels
    
    if class_counts:
        # Calculate weights: total_samples / class_count
        # Or simpler for BCE: (num_negatives / num_positives)
        # But we don't have total negatives easily. 
        # Heuristic: max_count / count usually works well to normalize.
        # Or just inverse frequency.
        counts = np.array(class_counts)
        # Avoid division by zero (shouldn't happen if vocab built from data)
        counts = np.maximum(counts, 1) 
        total = counts.sum()
        # weight = (total - count) / count
        weights = (total - counts) / counts
        pos_weight = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Using calculated class weights: {pos_weight.cpu().numpy()}")
    else:
        pos_weight = torch.ones([num_classes]).to(device) * 20.0
        print("Using default class weights: 20.0")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels, _, _, _ in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for features, labels, _, _, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                outputs = outputs.permute(0, 2, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def evaluate_model(model, val_loader, vocab):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for features, labels, _, _, _ in tqdm(val_loader, desc="Evaluation"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            
            probs_flat = probs.permute(0, 2, 1).reshape(-1, len(vocab)).cpu().numpy()
            labels_flat = labels.reshape(-1, len(vocab)).cpu().numpy()
            
            preds_bin = (probs_flat > 0.5).astype(int)
            
            all_preds.append(preds_bin)
            all_targets.append(labels_flat)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=vocab, zero_division=0))

def visualize_sample(model, dataset, vocab, save_path="supervised_visualization.png"):
    model.eval()
    idx = 0
    # Try to find a sample with labels
    for i in range(50):
        _, labels, _, _, _ = dataset[i]
        if labels.sum() > 0:
            idx = i
            break

    features, labels, vid, mouse_id, _ = dataset[idx]
    
    features_in = features.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(features_in)
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
        
    labels = labels.cpu().numpy().T
    
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

def main():
    # 1. Get Labeled Videos
    all_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    video_ids = [os.path.basename(f).replace(".parquet", "") for f in all_files if not f.endswith("_annotations.parquet")]
    
    labeled_vids = []
    for vid in video_ids:
        if os.path.exists(os.path.join(DATA_DIR, f"{vid}_annotations.parquet")):
            labeled_vids.append(vid)
            
    print(f"Found {len(labeled_vids)} labeled videos.")
    
    train_vids, val_vids = train_test_split(labeled_vids, test_size=0.1, random_state=42)
    
    # 2. Setup Datasets (Fits Scaler on Train)
    print("\nSetting up datasets...")
    # Initialize base to get vocab
    temp_ds = MouseBehaviorDataset(DATA_DIR, labeled_vids, is_training=False)
    vocab = temp_ds.behavior_vocab
    class_counts = temp_ds.class_counts
    print(f"Vocabulary ({len(vocab)}): {vocab}")
    
    # Train Dataset (Fits Scaler)
    base_train_ds = MouseBehaviorDataset(DATA_DIR, train_vids, vocab, seq_len=SEQ_LEN, is_training=True)
    scaler = base_train_ds.scaler
    train_ds = MouseIndexDataset(base_train_ds)
    
    # Val Dataset (Uses Train Scaler)
    base_val_ds = MouseBehaviorDataset(DATA_DIR, val_vids, vocab, seq_len=SEQ_LEN, is_training=False, scaler=scaler)
    val_ds = MouseIndexDataset(base_val_ds)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    # 3. Train
    print("\nStarting Training...")
    channel_sizes = [HIDDEN_CHANNELS] * 4
    model = TCN(num_inputs=base_train_ds.num_features, num_channels=channel_sizes, 
                kernel_size=KERNEL_SIZE, dropout=DROPOUT, num_classes=len(vocab)).to(device)
    
    train_model(model, train_loader, val_loader, EPOCHS, 'model_supervised.pth', class_counts=class_counts)
    
    # 4. Evaluate
    print("\nFinal Evaluation...")
    evaluate_model(model, val_loader, vocab)
    visualize_sample(model, val_ds, vocab)

if __name__ == "__main__":
    main()
