
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Configuration
DATA_DIR = './data/processed/train'
BATCH_SIZE = 1
SEQ_LEN = 100
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT = 0.2

# Check for GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# --- Class Definitions (copied from TCN.py) ---

class MouseBehaviorDataset(Dataset):
    def __init__(self, data_dir, video_ids, behavior_vocab=None, seq_len=100, is_training=True):
        self.data_dir = data_dir
        self.video_ids = video_ids
        self.seq_len = seq_len
        self.is_training = is_training
        
        # Load first video to get feature columns and behavior vocab if not provided
        self.feature_cols = self._get_feature_columns()
        if behavior_vocab is None:
            self.behavior_vocab = self._build_vocab()
        else:
            self.behavior_vocab = behavior_vocab
            
        self.num_classes = len(self.behavior_vocab)
        self.num_features = len(self.feature_cols)

    def _get_feature_columns(self):
        # Load a sample file to check columns
        sample_vid = self.video_ids[0]
        df = pd.read_parquet(os.path.join(self.data_dir, f"{sample_vid}.parquet"))
        
        # Exclude non-feature columns
        exclude = ['video_frame', 'mouse_id', 'video_id', 'lab_id']
        cols = [c for c in df.columns if c not in exclude]
        
        # Handle variable dist_to_mouse columns by ensuring we have a fixed set for up to 4 mice
        base_cols = [c for c in cols if not c.startswith('dist_to_mouse')]
        for i in range(1, 5):
            base_cols.append(f'dist_to_mouse_{i}')
            
        return base_cols

    def _build_vocab(self):
        behaviors = set()
        scan_ids = self.video_ids[:100]
        for vid in scan_ids:
            anno_path = os.path.join(self.data_dir, f"{vid}_annotations.parquet")
            if os.path.exists(anno_path):
                df = pd.read_parquet(anno_path)
                behaviors.update(df['action'].unique())
        return sorted(list(behaviors))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return None, None # Not used directly in wrapper

class MouseIndexDataset(Dataset):
    """Flattens (Video, Mouse) structure so each item is one mouse trajectory."""
    def __init__(self, data_dir, video_ids, behavior_vocab, seq_len=100, is_training=True):
        self.base_dataset = MouseBehaviorDataset(data_dir, video_ids, behavior_vocab, seq_len, is_training)
        self.indices = []
        
        # Pre-scan to build index (Video_Idx, Mouse_Idx)
        # print("Indexing mice...") 
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
        
        # Load features 
        feat_path = os.path.join(self.base_dataset.data_dir, f"{vid}.parquet")
        feat_df = pd.read_parquet(feat_path)
        
        for col in self.base_dataset.feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
                
        max_frame = feat_df['video_frame'].max()
        m_df = feat_df[feat_df['mouse_id'] == mouse_id].sort_values('video_frame')
        m_df = m_df.set_index('video_frame').reindex(range(int(max_frame) + 1), fill_value=0).reset_index()
        
        feats = m_df[self.base_dataset.feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats)
        
        # Load labels
        labels = np.zeros((len(feats), self.base_dataset.num_classes), dtype=np.float32)
        anno_path = os.path.join(self.base_dataset.data_dir, f"{vid}_annotations.parquet")
        if os.path.exists(anno_path):
            anno_df = pd.read_parquet(anno_path)
            mouse_annos = anno_df[anno_df['agent_id'] == mouse_id]
            for _, row in mouse_annos.iterrows():
                action = row['action']
                if action in self.base_dataset.behavior_vocab:
                    act_idx = self.base_dataset.behavior_vocab.index(action)
                    start = max(0, int(row['start_frame']))
                    stop = min(len(labels) - 1, int(row['stop_frame']))
                    labels[start:stop+1, act_idx] = 1.0

        # For testing, we don't random crop, we take a fixed window or the whole sequence if possible
        # But model expects fixed size or we iterate. 
        # Let's take the *middle* 100 frames for testing to see if any behavior is active
        if len(feats) > self.base_dataset.seq_len:
            start_idx = len(feats) // 2
            feats = feats[start_idx : start_idx + self.base_dataset.seq_len]
            labels = labels[start_idx : start_idx + self.base_dataset.seq_len]
        elif len(feats) < self.base_dataset.seq_len:
            pad_len = self.base_dataset.seq_len - len(feats)
            feats = np.pad(feats, ((0, pad_len), (0, 0)))
            labels = np.pad(labels, ((0, pad_len), (0, 0)))
            
        return torch.tensor(feats.T), torch.tensor(labels)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, 
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[2] != res.shape[2]:
            out = out[:, :, :res.shape[2]]
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

# --- Testing Logic ---

def test_categorization():
    print("Targeted Testing on Known Positive Examples...")
    
    # 1. Find examples for each behavior
    all_files = os.listdir(DATA_DIR)
    # To get vocab, we need some video ids. Let's just grab all parquet files first.
    all_videos = [f.replace('.parquet', '') for f in all_files 
                  if f.endswith('.parquet') and not f.endswith('annotations.parquet')]
    
    base_ds = MouseBehaviorDataset(DATA_DIR, all_videos)
    vocab = base_ds.behavior_vocab
    print(f"Behaviors: {vocab}")
    
    target_examples = {} # behavior -> {vid, start_frame, stop_frame, mouse_id}
    
    print("Scanning annotations for examples...")
    for vid in tqdm(all_videos):
        anno_path = os.path.join(DATA_DIR, f"{vid}_annotations.parquet")
        if not os.path.exists(anno_path):
            continue
            
        df = pd.read_parquet(anno_path)
        for behavior in vocab:
            if behavior not in target_examples:
                # Find row with this behavior
                row = df[df['action'] == behavior]
                if not row.empty:
                    # Pick the first one
                    r = row.iloc[0]
                    target_examples[behavior] = {
                        'vid': vid,
                        'start_frame': int(r['start_frame']),
                        'stop_frame': int(r['stop_frame']),
                        'mouse_id': int(r['agent_id'])
                    }
        
        # Stop if we found examples for all behaviors
        if len(target_examples) == len(vocab):
            break
            
    print(f"Found examples for: {list(target_examples.keys())}")
    
    # Load Model
    num_features = base_ds.num_features
    channel_sizes = [HIDDEN_CHANNELS] * 4
    model = TCN(num_inputs=num_features, 
                num_channels=channel_sizes, 
                kernel_size=KERNEL_SIZE, 
                dropout=DROPOUT, 
                num_classes=len(vocab)).to(device)
    
    model_path = 'tcn_mouse_behavior.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file '{model_path}' not found.")
        return
    model.eval()

    # 2. Run inference on each target
    print("\n--- Inference Results ---")
    
    for behavior, info in target_examples.items():
        vid = info['vid']
        mouse_id = info['mouse_id']
        # Target middle of the action
        center_frame = (info['start_frame'] + info['stop_frame']) // 2
        
        # Load features for this video
        feat_path = os.path.join(DATA_DIR, f"{vid}.parquet")
        feat_df = pd.read_parquet(feat_path)
        
        # Handle missing cols
        for col in base_ds.feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
                
        max_frame = feat_df['video_frame'].max()
        m_df = feat_df[feat_df['mouse_id'] == mouse_id].sort_values('video_frame')
        m_df = m_df.set_index('video_frame').reindex(range(int(max_frame) + 1), fill_value=0).reset_index()
        
        feats = m_df[base_ds.feature_cols].values.astype(np.float32)
        feats = np.nan_to_num(feats)
        
        # Construct window around center_frame
        # Start so that center_frame is in the middle of SEQ_LEN window
        start_idx = center_frame - (SEQ_LEN // 2)
        end_idx = start_idx + SEQ_LEN
        
        # Handle boundaries
        if start_idx < 0:
            start_idx = 0
            end_idx = SEQ_LEN
        if end_idx > len(feats):
            end_idx = len(feats)
            start_idx = max(0, end_idx - SEQ_LEN)
            
        window_feats = feats[start_idx:end_idx]
        
        # Pad if necessary (e.g. video shorter than seq_len)
        if len(window_feats) < SEQ_LEN:
            pad_len = SEQ_LEN - len(window_feats)
            window_feats = np.pad(window_feats, ((0, pad_len), (0, 0)))
            
        # Prepare tensor
        input_tensor = torch.tensor(window_feats.T).unsqueeze(0).to(device) # (1, Channels, Time)
        
        print(f"  Input Stats: Min={input_tensor.min().item():.2f}, Max={input_tensor.max().item():.2f}, Mean={input_tensor.mean().item():.2f}")
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            
        # Check prediction at the specific frame corresponding to center_frame
        # Relative index of center_frame in the window
        rel_idx = center_frame - start_idx
        if rel_idx >= SEQ_LEN: rel_idx = SEQ_LEN - 1
        
        # Get probs at this step
        step_probs = probs[0, :, rel_idx]
        step_logits = output[0, :, rel_idx]
        
        print(f"\nTarget Behavior: {behavior}")
        print(f"  Video: {vid}, Mouse: {mouse_id}, Frame: {center_frame}")
        print(f"  Window: {start_idx} - {end_idx}")
        
        # Print top predictions
        top_probs, top_indices = torch.topk(step_probs, k=3)
        print("  Predictions:")
        found_target = False
        for p, idx in zip(top_probs, top_indices):
            pred_behavior = vocab[idx]
            logit = step_logits[idx].item()
            print(f"    {pred_behavior}: Prob={p.item():.4f}, Logit={logit:.4f}")
            if pred_behavior == behavior and p.item() > 0.5:
                found_target = True
                
        if found_target:
            print("  Result: SUCCESS (Target detected)")
        else:
            # Check the probability of the target specifically
            target_idx = vocab.index(behavior)
            target_prob = step_probs[target_idx].item()
            target_logit = step_logits[target_idx].item()
            print(f"  Result: MISS (Target prob: {target_prob:.4f}, Logit: {target_logit:.4f})")


if __name__ == "__main__":
    test_categorization()
