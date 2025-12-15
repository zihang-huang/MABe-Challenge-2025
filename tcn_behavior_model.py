

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 32
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for the TCN model and training."""

    # Data paths
    DATA_DIR = Path("data/processed/train")
    OUTPUT_DIR = Path("outputs")

    # Feature columns (excluding metadata columns)
    FEATURE_COLS = [
        'ear_left_x', 'ear_left_y', 'ear_right_x', 'ear_right_y',
        'tail_base_x', 'tail_base_y', 'nose_x', 'nose_y',
        'body_elongation', 'velocity_x', 'velocity_y', 'speed',
        'acceleration', 'heading'
    ]

    # Model architecture
    # INPUT_DIM will be calculated dynamically
    NUM_CHANNELS = [64, 64, 128, 128, 256]  # TCN channel sizes
    KERNEL_SIZE = 5
    DROPOUT = 0.4

    # Training
    SEQUENCE_LENGTH = 120  # Increased temporal window size
    STRIDE = 32  # Stride for sliding window
    BATCH_SIZE = 1024
    NUM_WORKERS = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    
    # Imbalance Handling
    BACKGROUND_KEEP_PROB = 0.20  # Downsample 'no_behavior' to 20% during training

    # Evaluation
    VAL_SPLIT = 0.10
    TEST_SPLIT = 0.10

    # Evaluation
    VAL_SPLIT = 0.10
    TEST_SPLIT = 0.10



# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

class BehaviorDataset(Dataset):
    """Dataset for multi-agent behavior classification."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[np.ndarray]
    ):
        """
        Args:
            sequences: List of feature sequences [T, F]
            labels: List of label sequences [T]
        """
        if len(sequences) > 0:
            print(f"Converting {len(sequences)} sequences to tensors...")
            self.sequences = torch.FloatTensor(np.array(sequences))
            self.labels = torch.LongTensor(np.array(labels))
        else:
            self.sequences = torch.FloatTensor([])
            self.labels = torch.LongTensor([])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class DataProcessor:
    """Handles data loading and preprocessing for behavior classification."""

    def __init__(self, data_dir: Path, config: Config):
        self.data_dir = data_dir
        self.config = config
        self.scaler = StandardScaler()
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: Dict[int, str] = {}
        self.class_weights: Optional[torch.Tensor] = None

    def load_video_data(self, video_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Load features, annotations, and metadata for a video."""
        features_path = self.data_dir / f"{video_id}.parquet"
        annotations_path = self.data_dir / f"{video_id}_annotations.parquet"
        meta_path = self.data_dir / f"{video_id}_meta.json"

        features = pd.read_parquet(features_path)
        annotations = pd.read_parquet(annotations_path)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Handle missing values with interpolation
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].interpolate(
            method='linear', limit_direction='both').fillna(0)

        return features, annotations, metadata

    def create_frame_labels(
        self,
        features: pd.DataFrame,
        annotations: pd.DataFrame,
        num_frames: int
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Create per-frame labels for each agent-target pair."""
        pair_labels = {}

        for _, row in annotations.iterrows():
            agent_id = int(row['agent_id'])
            target_id = int(row['target_id'])
            action = row['action']
            start_frame = int(row['start_frame'])
            stop_frame = int(row['stop_frame'])

            key = (agent_id, target_id)
            if key not in pair_labels:
                pair_labels[key] = np.zeros(num_frames, dtype=np.int64)

            if action in self.action_to_idx:
                action_idx = self.action_to_idx[action]
                pair_labels[key][start_frame:stop_frame+1] = action_idx

        return pair_labels

    def extract_agent_target_features(
        self,
        features: pd.DataFrame,
        agent_id: int,
        target_id: int
    ) -> np.ndarray:
        """
        Extract combined features for an agent-target pair.
        Includes agent features, target features, and relative kinematics.
        """
        agent_data = features[features['mouse_id'] == agent_id].sort_values('video_frame')
        target_data = features[features['mouse_id'] == target_id].sort_values('video_frame')

        if len(agent_data) == 0 or len(target_data) == 0:
            return None

        # Extract basic features
        agent_features = agent_data[self.config.FEATURE_COLS].values
        target_features = target_data[self.config.FEATURE_COLS].values

        # Relative position
        agent_nose = agent_data[['nose_x', 'nose_y']].values
        target_nose = target_data[['nose_x', 'nose_y']].values
        rel_pos = agent_nose - target_nose
        distance = np.sqrt(np.sum(rel_pos**2, axis=1, keepdims=True))

        # Relative heading
        agent_heading = agent_data['heading'].values.reshape(-1, 1)
        target_heading = target_data['heading'].values.reshape(-1, 1)
        rel_heading = agent_heading - target_heading

        # Relative velocity
        agent_vel = agent_data[['velocity_x', 'velocity_y']].values
        target_vel = target_data[['velocity_x', 'velocity_y']].values
        rel_vel = agent_vel - target_vel
        rel_speed = np.sqrt(np.sum(rel_vel**2, axis=1, keepdims=True))

        # Body elongation change (derivative)
        # We compute this simply as diff, prepending 0 for the first frame
        agent_elongation = agent_data['body_elongation'].values
        agent_elongation_change = np.diff(agent_elongation, prepend=agent_elongation[0]).reshape(-1, 1)

        # Combine all features
        combined_features = np.concatenate([
            agent_features,      # Agent pose + kinematics
            target_features,     # Target pose + kinematics
            rel_pos,             # Relative position (x, y)
            distance,            # Distance
            rel_heading,         # Relative heading
            rel_vel,             # Relative velocity (x, y)
            rel_speed,           # Relative speed
            agent_elongation_change # Change in body elongation
        ], axis=1)

        return combined_features

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        downsample: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create fixed-length sequences using sliding window.
        Optionally downsample purely background sequences.
        """
        sequences = []
        label_sequences = []
        num_frames = len(features)

        for start in range(0, num_frames - self.config.SEQUENCE_LENGTH + 1, self.config.STRIDE):
            end = start + self.config.SEQUENCE_LENGTH
            seq = features[start:end]
            label_seq = labels[start:end]

            # Quality check: Skip sequences with too many NaNs
            if np.isnan(seq).sum() / seq.size > 0.3:
                continue

            # Downsampling: If sequence is purely background (0), keep with low probability
            if downsample and np.all(label_seq == 0):
                if np.random.random() > self.config.BACKGROUND_KEEP_PROB:
                    continue

            seq = np.nan_to_num(seq, nan=0.0)
            sequences.append(seq)
            label_sequences.append(label_seq)

        return sequences, label_sequences

    def prepare_classes(self, video_files: List[str]) -> None:
        """Scan all annotations to build the class map."""
        print("Scanning classes...")
        all_actions = set()
        for video_id in tqdm(video_files, desc="Collecting actions"):
            try:
                annotations_path = self.data_dir / f"{video_id}_annotations.parquet"
                if annotations_path.exists():
                    annotations = pd.read_parquet(annotations_path)
                    all_actions.update(annotations['action'].unique())
            except Exception:
                continue

        all_actions = sorted(list(all_actions))
        self.action_to_idx = {action: idx + 1 for idx, action in enumerate(all_actions)}
        self.action_to_idx['no_behavior'] = 0
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        print(f"Found {len(all_actions)} behavior classes (+ no_behavior)")

    def process_dataset(
        self,
        video_ids: List[str],
        is_training: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Counter]:
        """
        Process a specific list of videos.
        
        Args:
            video_ids: List of video IDs to process
            is_training: If True, fits scaler and downsamples background
            
        Returns:
            sequences, labels, label_counts
        """
        all_sequences = []
        all_labels = []
        label_counts = Counter()

        desc = "Processing Train" if is_training else "Processing Eval"
        
        for video_id in tqdm(video_ids, desc=desc):
            try:
                features, annotations, _ = self.load_video_data(video_id)
                frames = features['video_frame'].unique()
                mouse_ids = features['mouse_id'].unique()
                
                pair_labels = self.create_frame_labels(features, annotations, len(frames))

                for agent_id in mouse_ids:
                    for target_id in mouse_ids:
                        if agent_id == target_id: continue
                        
                        # Note: We process all pairs, but only those with annotations
                        # or valid interactions will effectively have labels.
                        # If a pair has no annotations, pair_labels[key] creates 0s.
                        key = (agent_id, target_id)
                        
                        # Optimization: Skip pairs that definitely have no interaction 
                        # if we are just looking for behaviors. 
                        # But for TCN we need context. 
                        # Current logic: If create_frame_labels made an entry, use it.
                        if key not in pair_labels:
                            continue

                        combined_features = self.extract_agent_target_features(
                            features, agent_id, target_id
                        )

                        if combined_features is None:
                            continue

                        seqs, labs = self.create_sequences(
                            combined_features,
                            pair_labels[key],
                            downsample=is_training
                        )
                        
                        if not seqs:
                            continue
                            
                        # Online scaler fitting (only during training)
                        if is_training:
                            flat_seqs = np.concatenate(seqs, axis=0)
                            self.scaler.partial_fit(flat_seqs)

                        all_sequences.extend(seqs)
                        all_labels.extend(labs)
                        
                        # Count labels (for weighting)
                        for lab in labs:
                            label_counts.update(lab.tolist())

            except Exception as e:
                # print(f"Error processing video {video_id}: {e}")
                continue

        # Transform all sequences
        # Note: For training, we fit above. For val/test, we assume scaler is already fit.
        if all_sequences:
            print(f"Scaling {len(all_sequences)} sequences...")
            for i in range(len(all_sequences)):
                shape = all_sequences[i].shape
                # Reshape to [N, F] for scaler, then back to [T, F]
                # Check for infinite values before scaling
                if not np.isfinite(all_sequences[i]).all():
                     all_sequences[i] = np.nan_to_num(all_sequences[i], nan=0.0, posinf=0.0, neginf=0.0)

                scaled = self.scaler.transform(all_sequences[i].reshape(-1, shape[-1]))
                all_sequences[i] = scaled.reshape(shape)

        return all_sequences, all_labels, label_counts

    def compute_class_weights(self, label_counts: Counter) -> torch.Tensor:
        """
        Compute class weights using 'balanced' strategy:
        n_samples / (n_classes * np.bincount(y))
        """
        total_samples = sum(label_counts.values())
        num_classes = len(self.action_to_idx)
        
        # Sort counts by class index to ensure correct order
        counts = np.array([label_counts.get(i, 0) for i in range(num_classes)])
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        
        # Calculate balanced weights
        weights = total_samples / (num_classes * counts)
        
        # Normalize weights so mean is 1.0 (keeps loss magnitude similar to unweighted)
        weights = weights / weights.mean()
        
        print("Class weights (Balanced & Normalized):")
        for i, weight in enumerate(weights):
            if i < 10:
                print(f"  Class {i}: {weight:.4f} (count={counts[i]})")
                
        return torch.FloatTensor(weights).to(DEVICE)


# =============================================================================
# TCN Model Architecture
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding to maintain sequence length."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolutions and residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]
        out = self.network(x)
        return out.transpose(1, 2)  # [B, F, T] -> [B, T, F]


class BehaviorClassifier(nn.Module):
    """Complete model for behavior classification."""
    def __init__(self, input_dim, num_classes, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tcn = TemporalConvNet(num_channels[0], num_channels, kernel_size, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.tcn(x)
        logits = self.classifier(x)
        return logits


# =============================================================================
# Training and Evaluation
# =============================================================================

class Trainer:
    """Handles model training and evaluation."""
    def __init__(self, model, config, class_weights=None):
        self.model = model.to(DEVICE)
        self.config = config

        # Use Standard Cross Entropy with Label Smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=0.1,
            ignore_index=-1
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
        self.scaler = GradScaler()
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_f1': [], 'val_f1': [], 
            'learning_rates': []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            with autocast():
                logits = self.model(sequences)
                B, T, C = logits.shape
                loss = self.criterion(logits.view(B * T, C), labels.view(B * T))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return avg_loss, f1

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            logits = self.model(sequences)
            B, T, C = logits.shape
            loss = self.criterion(logits.view(B * T, C), labels.view(B * T))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return avg_loss, f1, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader, val_loader, num_epochs):
        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            train_loss, train_f1 = self.train_epoch(train_loader)
            
            if (epoch + 1) % 5 == 0:
                val_loss, val_f1, _, _ = self.evaluate(val_loader)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    print(f"New best model! Val F1: {best_val_f1:.4f}")
            else:
                val_loss, val_f1 = np.nan, np.nan

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            if not np.isnan(val_loss):
                print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self.history


# =============================================================================
# Visualization
# =============================================================================

class Visualizer:
    """Handles visualization of results."""
    def __init__(self, output_dir: Path, idx_to_action: Dict[int, str]):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.idx_to_action = idx_to_action

    def plot_training_history(self, history: Dict) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Filter NaNs for plotting
        val_loss = [x for x in history['val_loss'] if not np.isnan(x)]
        val_f1 = [x for x in history['val_f1'] if not np.isnan(x)]
        val_epochs = [i for i, x in enumerate(history['val_loss']) if not np.isnan(x)]

        axes[0].plot(history['train_loss'], label='Train')
        if val_loss:
            axes[0].plot(val_epochs, val_loss, label='Validation', marker='o')
        axes[0].set_title('Training Loss')
        axes[0].legend()

        axes[1].plot(history['train_f1'], label='Train')
        if val_f1:
            axes[1].plot(val_epochs, val_f1, label='Validation', marker='o')
        axes[1].set_title('Macro F1 Score')
        axes[1].legend()

        axes[2].plot(history['learning_rates'])
        axes[2].set_title('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()

    def save_classification_report(self, y_true, y_pred) -> str:
        unique_labels = sorted(set(y_true) | set(y_pred))
        label_names = [self.idx_to_action.get(i, f'class_{i}') for i in unique_labels]
        report = classification_report(
            y_true, y_pred, labels=unique_labels, target_names=label_names, zero_division=0
        )
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("TCN Behavior Classification Model - Improved")
    print("MABe Challenge 2025")
    print("=" * 60)

    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processor = DataProcessor(config.DATA_DIR, config)

    # 1. Get all videos and prepare classes
    video_files = [f.stem for f in config.DATA_DIR.glob("*.parquet") if '_' not in f.stem]
    if not video_files:
        print("No data found!")
        return

    processor.prepare_classes(video_files)
    num_classes = len(processor.action_to_idx)
    
    # 2. Split videos FIRST
    print("\n[Split] Splitting videos into Train/Val/Test...")
    train_vids, test_vids = train_test_split(video_files, test_size=config.VAL_SPLIT + config.TEST_SPLIT, random_state=SEED)
    val_vids, test_vids = train_test_split(test_vids, test_size=0.5, random_state=SEED) # Equal split for val/test
    
    print(f"Train videos: {len(train_vids)}")
    print(f"Val videos:   {len(val_vids)}")
    print(f"Test videos:  {len(test_vids)}")

    # 3. Process datasets independently
    # Train: Fit scaler, downsample background
    print("\n[Train Data] Processing...")
    train_seqs, train_labels, train_counts = processor.process_dataset(train_vids, is_training=True)
    
    # Val/Test: Transform scaler, keep all data
    print("\n[Val Data] Processing...")
    val_seqs, val_labels, _ = processor.process_dataset(val_vids, is_training=False)
    
    print("\n[Test Data] Processing...")
    test_seqs, test_labels, _ = processor.process_dataset(test_vids, is_training=False)

    if len(train_seqs) == 0:
        print("Error: No training data generated.")
        return

    # 4. Create Datasets and Loaders
    train_dataset = BehaviorDataset(train_seqs, train_labels)
    val_dataset = BehaviorDataset(val_seqs, val_labels)
    test_dataset = BehaviorDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 5. Initialize Model
    input_dim = train_seqs[0].shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    model = BehaviorClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=config.NUM_CHANNELS,
        kernel_size=config.KERNEL_SIZE,
        dropout=config.DROPOUT
    )

    # 6. Train
    print("\nStarting Training...")
    # Calculate weights based on the downsampled training data
    # class_weights = processor.compute_class_weights(train_counts)
    print("WARNING: Disabling class weights to debug exploding loss.")
    class_weights = None
    
    trainer = Trainer(model, config, class_weights)
    history = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)

    # 7. Save
    model_path = config.OUTPUT_DIR / 'tcn_behavior_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'action_to_idx': processor.action_to_idx
    }, model_path)
    
    # 8. Evaluate
    print("\nFinal Evaluation on Test Set...")
    _, test_f1, test_preds, test_true = trainer.evaluate(test_loader)
    print(f"Test Macro F1: {test_f1:.4f}")
    
    visualizer = Visualizer(config.OUTPUT_DIR, processor.idx_to_action)
    visualizer.plot_training_history(history)
    print(visualizer.save_classification_report(test_true, test_preds))


if __name__ == "__main__":
    main()
