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
from tqdm import tqdm
import pickle

# Configuration
DATA_DIR = './data/processed/train'
PSEUDO_DIR = './data/processed/train_pseudo'
BATCH_SIZE = 64
SEQ_LEN = 100
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2
LR = 0.001
CONFIDENCE_THRESHOLD = 0.20
INITIAL_EPOCHS = 20
FINAL_EPOCHS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(PSEUDO_DIR, exist_ok=True)

# --- Dataset Classes ---

class MouseBehaviorDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab=None, seq_len=100, is_training=True, pseudo_dir=None):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.seq_len = seq_len
        self.is_training = is_training
        self.pseudo_dir = pseudo_dir
        
        self.feature_cols = self._get_feature_columns()
        if behavior_vocab is None:
            self.behavior_vocab = self._build_vocab()
        else:
            self.behavior_vocab = behavior_vocab
            
        self.num_classes = len(self.behavior_vocab)
        self.num_features = len(self.feature_cols)
        self.scaler = self._get_scaler()

    def _get_feature_columns(self):
        # Determine feature columns from the first available video
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
        # Scan labeled videos for vocabulary
        scan_ids = [vid for vid in self.video_ids if os.path.exists(os.path.join(self.data_dir, f"{vid}_annotations.parquet"))]
        scan_ids = scan_ids[:100] # Limit scan
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
        
        if self.is_training:
            print("Fitting scaler...")
            scaler = StandardScaler()
            sample_size = min(len(self.video_ids), 200)
            sample_vids = np.random.choice(self.video_ids, sample_size, replace=False) if self.video_ids else []
            all_feats = []
            for vid in tqdm(sample_vids, desc="Scaler Data"):
                feat_path = os.path.join(self.data_dir, f"{vid}.parquet")
                if os.path.exists(feat_path):
                    df = pd.read_parquet(feat_path)
                    for col in self.feature_cols:
                        if col not in df.columns: df[col] = 0.0
                    feats = df[self.feature_cols].values.astype(np.float32)
                    feats = np.nan_to_num(feats)
                    all_feats.append(feats[::10])
            
            if all_feats:
                concat_feats = np.concatenate(all_feats, axis=0)
                scaler.fit(concat_feats)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                return scaler
        return None

class MouseIndexDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab, seq_len=100, is_training=True, pseudo_dir=None):
        self.base_dataset = MouseBehaviorDataset(data_dir, video_ids, behavior_vocab, seq_len, is_training, pseudo_dir)
        self.indices = []
        
        print(f"Indexing {len(video_ids)} videos...")
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
            
        # Load Pseudo Annotations if available
        if self.base_dataset.pseudo_dir:
            pseudo_path = os.path.join(self.base_dataset.pseudo_dir, f"{vid}_annotations.parquet")
            if os.path.exists(pseudo_path):
                 self._fill_labels(labels, pseudo_path, mouse_id)

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

def get_video_lists(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    video_ids = [os.path.basename(f).replace(".parquet", "") for f in all_files if not f.endswith("_annotations.parquet")]
    
    labeled = []
    unlabeled = []
    
    for vid in video_ids:
        if os.path.exists(os.path.join(data_dir, f"{vid}_annotations.parquet")):
            labeled.append(vid)
        else:
            unlabeled.append(vid)
            
    return labeled, unlabeled

def train_model(model, train_loader, val_loader, epochs, save_path):
    # Handle class imbalance by weighting positive classes
    # Get num_classes from the model's last layer
    num_classes = model.classifier.out_channels
    pos_weight = torch.ones([num_classes]).to(device) * 20.0
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

def generate_pseudo_labels(model, unlabeled_loader, vocab, threshold=0.95):
    model.eval()
    print("Generating pseudo-labels...")
    
    # Cache predictions per video to aggregate across mice
    # Dictionary: video_id -> list of DataFrames (one per mouse)
    predictions = {}
    
    global_max_prob = 0.0
    
    with torch.no_grad():
        for i, (features, _, vid_tensor, mouse_id_tensor, max_frame_tensor) in tqdm(enumerate(unlabeled_loader), total=len(unlabeled_loader), desc="Inference"):
            features = features.to(device)
            # vid_tensor and mouse_id_tensor are batched, but we use batch_size=1 for simplicity in logic or handle batch
            # Assuming batch_size=1 for inference to handle full sequences easily or we need to handle variable lengths
            # The Dataset currently windows or pads. For inference we want full sequence.
            # However, the dataset implementation enforces SEQ_LEN windowing/padding even for validation/test if is_training=False?
            # Let's check Dataset. No, if is_training=False, it still windows/pads?
            # Ah, the dataset pads if len < seq_len. 
            # If len > seq_len and is_training=False, it DOES NOT window, it returns full sequence! Correct.
            
            outputs = model(features) # (B, Classes, Time)
            probs = torch.sigmoid(outputs) # (B, Classes, Time)
            
            # Track max prob for debugging
            current_max = probs.max().item()
            if current_max > global_max_prob:
                global_max_prob = current_max
            
            if i % 1000 == 0:
                print(f"  Step {i}: Max Prob seen so far: {global_max_prob:.4f}")

            probs = probs.permute(0, 2, 1).cpu().numpy() # (B, Time, Classes)
            
            vid = vid_tensor[0] # Tuple from DataLoader collate? No, list of strings
            mouse_id = mouse_id_tensor[0].item()
            
            # Filter high confidence
            # For each frame, if prob > threshold, record action
            # Result: list of (start, stop, action)
            
            pred_frames = probs[0] # (Time, Classes)
            
            # Simple frame-by-frame extraction
            # Optimization: could use vectorization, but loop is fine for now
            
            actions_list = []
            
            for t in range(len(pred_frames)):
                for c_idx, prob in enumerate(pred_frames[t]):
                    if prob > threshold:
                        actions_list.append({
                            'agent_id': mouse_id,
                            'target_id': -1, # Unknown target logic for now, or assume single agent action
                            'action': vocab[c_idx],
                            'start_frame': t,
                            'stop_frame': t
                        })
                        
            if actions_list:
                df = pd.DataFrame(actions_list)
                # Consolidate consecutive frames
                # Sort by action, then frame
                df = df.sort_values(['action', 'start_frame'])
                
                consolidated = []
                if not df.empty:
                    for action, group in df.groupby('action'):
                        group = group.sort_values('start_frame')
                        start = group.iloc[0]['start_frame']
                        prev = start
                        for _, row in group.iterrows():
                            curr = row['start_frame']
                            if curr > prev + 1:
                                consolidated.append([mouse_id, -1, action, start, prev])
                                start = curr
                            prev = curr
                        consolidated.append([mouse_id, -1, action, start, prev])
                
                con_df = pd.DataFrame(consolidated, columns=['agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
                
                if vid not in predictions:
                    predictions[vid] = []
                predictions[vid].append(con_df)

    # Save to parquet
    count = 0
    for vid, dfs in predictions.items():
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            save_path = os.path.join(PSEUDO_DIR, f"{vid}_annotations.parquet")
            full_df.to_parquet(save_path)
            count += 1
            
    print(f"Generated pseudo-labels for {count} videos.")

# --- Main ---

def main():
    labeled_vids, unlabeled_vids = get_video_lists(DATA_DIR)
    print(f"Found {len(labeled_vids)} labeled and {len(unlabeled_vids)} unlabeled videos.")
    
    # Split labeled for validation
    train_vids, val_vids = train_test_split(labeled_vids, test_size=0.1, random_state=42)
    
    # 1. Initial Training
    print("\n--- Stage 1: Initial Training ---")
    base_ds = MouseBehaviorDataset(DATA_DIR, labeled_vids)
    vocab = base_ds.behavior_vocab
    print(f"Vocabulary ({len(vocab)}): {vocab}")
    
    train_ds = MouseIndexDataset(DATA_DIR, train_vids, vocab, seq_len=SEQ_LEN, is_training=True)
    val_ds = MouseIndexDataset(DATA_DIR, val_vids, vocab, seq_len=SEQ_LEN, is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    channel_sizes = [HIDDEN_CHANNELS] * 4
    model = TCN(num_inputs=base_ds.num_features, num_channels=channel_sizes, 
                kernel_size=KERNEL_SIZE, dropout=DROPOUT, num_classes=len(vocab)).to(device)
    
    train_model(model, train_loader, val_loader, INITIAL_EPOCHS, 'model_initial.pth')
    
    # 2. Pseudo-Labeling
    print("\n--- Stage 2: Pseudo-Labeling ---")
    
    unlabeled_ds = MouseIndexDataset(DATA_DIR, unlabeled_vids, vocab, seq_len=SEQ_LEN, is_training=False)
    # Batch size 1 for inference to handle full sequences
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=1, shuffle=False, num_workers=0)
    
    generate_pseudo_labels(model, unlabeled_loader, vocab, threshold=CONFIDENCE_THRESHOLD)
    
    # 3. Retraining
    print("\n--- Stage 3: Retraining with Pseudo-Labels ---")
    
    # Use all labeled + pseudo-labeled
    # Pseudo-labeled videos are those in PSEUDO_DIR
    pseudo_vids = [os.path.basename(f).replace("_annotations.parquet", "") for f in glob.glob(os.path.join(PSEUDO_DIR, "*.parquet"))]
    combined_vids = train_vids + pseudo_vids
    
    print(f"Training on {len(train_vids)} real + {len(pseudo_vids)} pseudo-labeled videos.")
    
    # New dataset with pseudo_dir
    combined_train_ds = MouseIndexDataset(DATA_DIR, combined_vids, vocab, seq_len=SEQ_LEN, 
                                          is_training=True, pseudo_dir=PSEUDO_DIR)
    
    combined_loader = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Re-initialize model or fine-tune? Usually fine-tune or re-train. Let's fine-tune.
    # To re-train from scratch, uncomment:
    # model = TCN(...)
    
    train_model(model, combined_loader, val_loader, FINAL_EPOCHS, 'model_final.pth')

if __name__ == "__main__":
    main()
