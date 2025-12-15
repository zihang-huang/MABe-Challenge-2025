"""
Temporal Convolutional Network for Multi-Agent Mouse Behavior Classification
MABe Challenge 2025

This module implements a TCN-based model for classifying social and non-social
behaviors in mice based on pose tracking data.
"""

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
from sklearn.preprocessing import LabelEncoder

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

    # Distance features will be added dynamically based on target
    DISTANCE_COLS_TEMPLATE = ['dist_to_mouse_1', 'dist_to_mouse_2',
                              'dist_to_mouse_3', 'dist_to_mouse_4']

    # Model architecture
    INPUT_DIM = 32  # Will be calculated: agent_features + target_features + relative_features
    NUM_CHANNELS = [64, 64, 128, 128, 256]  # TCN channel sizes
    KERNEL_SIZE = 5
    DROPOUT = 0.3

    # Training
    SEQUENCE_LENGTH = 64  # Temporal window size
    STRIDE = 32  # Stride for sliding window
    BATCH_SIZE = 2048  # Increased for RTX 5090
    NUM_WORKERS = 8  # Parallel data loading
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

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
        labels: List[np.ndarray],
        sample_weights: Optional[List[float]] = None
    ):
        """
        Args:
            sequences: List of feature sequences [T, F]
            labels: List of label sequences [T]
            sample_weights: Optional sample weights for imbalanced data
        """
        # Convert to tensors immediately to save CPU time during training
        print("Converting dataset to tensors...")
        self.sequences = torch.FloatTensor(np.array(sequences))
        self.labels = torch.LongTensor(np.array(labels))
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class DataProcessor:
    """Handles data loading and preprocessing for behavior classification."""

    def __init__(self, data_dir: Path, config: Config):
        self.data_dir = data_dir
        self.config = config
        self.label_encoder = LabelEncoder()
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

        return features, annotations, metadata

    def create_frame_labels(
        self,
        features: pd.DataFrame,
        annotations: pd.DataFrame,
        num_frames: int
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Create per-frame labels for each agent-target pair.

        Returns:
            Dictionary mapping (agent_id, target_id) to frame-level labels
        """
        # Initialize with "no_behavior" (index 0)
        pair_labels = {}

        for _, row in annotations.iterrows():
            agent_id = int(row['agent_id'])
            target_id = int(row['target_id'])
            action = row['action']
            start_frame = int(row['start_frame'])
            stop_frame = int(row['stop_frame'])

            key = (agent_id, target_id)
            if key not in pair_labels:
                # Initialize with zeros (no behavior)
                pair_labels[key] = np.zeros(num_frames, dtype=np.int64)

            # Assign action label to frames
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

        Features include:
        - Agent pose and kinematic features
        - Target pose and kinematic features
        - Relative features (distance, relative position, etc.)
        """
        # Get agent and target data
        agent_data = features[features['mouse_id'] == agent_id].sort_values('video_frame')
        target_data = features[features['mouse_id'] == target_id].sort_values('video_frame')

        if len(agent_data) == 0 or len(target_data) == 0:
            return None

        # Align by frame
        frames = agent_data['video_frame'].values

        # Extract agent features
        agent_features = agent_data[self.config.FEATURE_COLS].values

        # Extract target features
        target_features = target_data[self.config.FEATURE_COLS].values

        # Calculate relative features
        agent_nose = agent_data[['nose_x', 'nose_y']].values
        target_nose = target_data[['nose_x', 'nose_y']].values

        # Relative position
        rel_pos = agent_nose - target_nose

        # Distance between agents
        distance = np.sqrt(np.sum(rel_pos**2, axis=1, keepdims=True))

        # Relative heading
        agent_heading = agent_data['heading'].values.reshape(-1, 1)
        target_heading = target_data['heading'].values.reshape(-1, 1)
        rel_heading = agent_heading - target_heading

        # Combine all features
        combined_features = np.concatenate([
            agent_features,      # Agent pose + kinematics
            target_features,     # Target pose + kinematics
            rel_pos,             # Relative position (x, y)
            distance,            # Distance
            rel_heading          # Relative heading
        ], axis=1)

        return combined_features

    def create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_length: int,
        stride: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Create fixed-length sequences using sliding window."""
        sequences = []
        label_sequences = []

        num_frames = len(features)

        for start in range(0, num_frames - seq_length + 1, stride):
            end = start + seq_length
            seq = features[start:end]
            label_seq = labels[start:end]

            # Skip sequences with too many NaN values
            if np.isnan(seq).sum() / seq.size < 0.3:
                # Fill remaining NaNs with interpolation or zeros
                seq = np.nan_to_num(seq, nan=0.0)
                sequences.append(seq)
                label_sequences.append(label_seq)

        return sequences, label_sequences

    def process_all_videos(
        self,
        max_videos: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Process all videos and create training sequences.

        Returns:
            sequences: List of feature sequences
            labels: List of label sequences
            video_ids: List of source video IDs
        """
        # Get all video IDs
        video_files = [f.stem for f in self.data_dir.glob("*.parquet")
                       if '_' not in f.stem]

        if max_videos:
            video_files = video_files[:max_videos]

        print(f"Processing {len(video_files)} videos...")

        # First pass: collect all actions
        all_actions = set()
        for video_id in tqdm(video_files, desc="Collecting actions"):
            try:
                annotations_path = self.data_dir / f"{video_id}_annotations.parquet"
                if annotations_path.exists():
                    annotations = pd.read_parquet(annotations_path)
                    all_actions.update(annotations['action'].unique())
            except Exception as e:
                continue

        # Create action mapping (0 = no_behavior)
        all_actions = sorted(list(all_actions))
        self.action_to_idx = {action: idx + 1 for idx, action in enumerate(all_actions)}
        self.action_to_idx['no_behavior'] = 0
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

        print(f"Found {len(all_actions)} behavior classes (+ no_behavior)")

        # Second pass: create sequences
        all_sequences = []
        all_labels = []
        all_video_ids = []
        label_counts = Counter()

        for video_id in tqdm(video_files, desc="Processing videos"):
            try:
                features, annotations, metadata = self.load_video_data(video_id)

                # Get unique frames and mouse IDs
                frames = features['video_frame'].unique()
                num_frames = len(frames)
                mouse_ids = features['mouse_id'].unique()

                # Create frame labels for each pair
                pair_labels = self.create_frame_labels(features, annotations, num_frames)

                # Process each agent-target pair
                for agent_id in mouse_ids:
                    for target_id in mouse_ids:
                        key = (agent_id, target_id)

                        # Only process pairs that have annotations
                        if key not in pair_labels:
                            continue

                        # Extract features
                        combined_features = self.extract_agent_target_features(
                            features, agent_id, target_id
                        )

                        if combined_features is None:
                            continue

                        # Create sequences
                        seqs, labs = self.create_sequences(
                            combined_features,
                            pair_labels[key],
                            self.config.SEQUENCE_LENGTH,
                            self.config.STRIDE
                        )

                        all_sequences.extend(seqs)
                        all_labels.extend(labs)
                        all_video_ids.extend([video_id] * len(seqs))

                        # Count labels
                        for lab in labs:
                            label_counts.update(lab.tolist())

            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue

        print(f"Created {len(all_sequences)} sequences")
        print(f"Label distribution: {dict(label_counts.most_common(10))}...")

        # Calculate class weights for imbalanced data
        total_samples = sum(label_counts.values())
        num_classes = len(self.action_to_idx)
        class_weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(min(weight, 10.0))  # Cap weights
        self.class_weights = torch.FloatTensor(class_weights).to(DEVICE)

        return all_sequences, all_labels, all_video_ids


# =============================================================================
# TCN Model Architecture
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding to maintain sequence length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove future padding to maintain causality
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated causal convolutions and residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.

    Uses dilated causal convolutions with exponentially increasing
    dilation rates to capture long-range temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, F] (batch, time, features)

        Returns:
            Output tensor [B, T, C] (batch, time, channels)
        """
        # TCN expects [B, C, T] format
        x = x.transpose(1, 2)
        out = self.network(x)
        # Return to [B, T, C] format
        return out.transpose(1, 2)


class BehaviorClassifier(nn.Module):
    """
    Complete model for behavior classification.

    Combines TCN for temporal feature extraction with
    classification heads for per-frame predictions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, num_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal convolutional network
        self.tcn = TemporalConvNet(
            num_channels[0], num_channels, kernel_size, dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, F]

        Returns:
            Logits tensor [B, T, num_classes]
        """
        # Project input
        x = self.input_proj(x)

        # Temporal modeling
        x = self.tcn(x)

        # Per-frame classification
        logits = self.classifier(x)

        return logits


# =============================================================================
# Training and Evaluation
# =============================================================================

class Trainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(DEVICE)
        self.config = config

        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-1  # For masked positions
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast():
                logits = self.model(sequences)

                # Compute loss (reshape for cross-entropy)
                B, T, C = logits.shape
                loss = self.criterion(
                    logits.view(B * T, C),
                    labels.view(B * T)
                )

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Collect predictions
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, f1

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            logits = self.model(sequences)

            # Compute loss
            B, T, C = logits.shape
            loss = self.criterion(
                logits.view(B * T, C),
                labels.view(B * T)
            )
            total_loss += loss.item()

            # Collect predictions
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, f1, np.array(all_preds), np.array(all_labels)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict:
        """Full training loop."""
        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_f1 = self.train_epoch(train_loader)

            # Evaluate (every 25 epochs)
            if (epoch + 1) % 25 == 0:
                val_loss, val_f1, _, _ = self.evaluate(val_loader)

                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict().copy()
                    print(f"New best model! Val F1: {best_val_f1:.4f}")
            else:
                val_loss, val_f1 = np.nan, np.nan

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            if not np.isnan(val_loss):
                print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

        # Restore best model
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
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1 Score
        axes[1].plot(history['train_f1'], label='Train')
        axes[1].plot(history['val_f1'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Macro F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(history['learning_rates'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        plt.close()
        print(f"Saved training history plot to {self.output_dir / 'training_history.png'}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True
    ) -> None:
        """Plot confusion matrix."""
        # Get unique labels present in data
        unique_labels = sorted(set(y_true) | set(y_pred))
        label_names = [self.idx_to_action.get(i, f'class_{i}') for i in unique_labels]

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            xticklabels=label_names, yticklabels=label_names,
            cmap='Blues', square=True
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        print(f"Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")

    def plot_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Plot per-class precision, recall, and F1."""
        unique_labels = sorted(set(y_true) | set(y_pred))
        label_names = [self.idx_to_action.get(i, f'class_{i}') for i in unique_labels]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, zero_division=0
        )

        # Create DataFrame for plotting
        metrics_df = pd.DataFrame({
            'Class': label_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })

        # Sort by F1 score
        metrics_df = metrics_df.sort_values('F1-Score', ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Bar plot of metrics
        x = np.arange(len(metrics_df))
        width = 0.25

        axes[0].barh(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        axes[0].barh(x, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        axes[0].barh(x + width, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(metrics_df['Class'])
        axes[0].set_xlabel('Score')
        axes[0].set_title('Per-Class Metrics')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='x')

        # Support distribution
        axes[1].barh(x, metrics_df['Support'], alpha=0.8, color='green')
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(metrics_df['Class'])
        axes[1].set_xlabel('Number of Samples')
        axes[1].set_title('Class Distribution (Support)')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=150)
        plt.close()
        print(f"Saved per-class metrics to {self.output_dir / 'per_class_metrics.png'}")

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Generate and save classification report."""
        unique_labels = sorted(set(y_true) | set(y_pred))
        label_names = [self.idx_to_action.get(i, f'class_{i}') for i in unique_labels]

        report = classification_report(
            y_true, y_pred,
            labels=unique_labels,
            target_names=label_names,
            zero_division=0
        )

        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)

        print(f"Saved classification report to {report_path}")
        return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("TCN Behavior Classification Model")
    print("MABe Challenge 2025")
    print("=" * 60)

    config = Config()

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Data Processing
    # ==========================================================================
    print("\n[1/5] Loading and processing data...")

    processor = DataProcessor(config.DATA_DIR, config)

    # Process videos (limit for faster testing, remove for full training)
    sequences, labels, video_ids = processor.process_all_videos()

    if len(sequences) == 0:
        print("No sequences created. Check data paths and format.")
        return

    # Get input dimension from data
    input_dim = sequences[0].shape[1]
    num_classes = len(processor.action_to_idx)

    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Total sequences: {len(sequences)}")

    # ==========================================================================
    # Train/Val/Test Split
    # ==========================================================================
    print("\n[2/5] Splitting data...")

    # Split by video to prevent data leakage
    unique_videos = list(set(video_ids))
    train_videos, temp_videos = train_test_split(
        unique_videos,
        test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        random_state=SEED
    )
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=SEED
    )

    # Create split indices
    train_indices = [i for i, vid in enumerate(video_ids) if vid in train_videos]
    val_indices = [i for i, vid in enumerate(video_ids) if vid in val_videos]
    test_indices = [i for i, vid in enumerate(video_ids) if vid in test_videos]

    print(f"Train: {len(train_indices)} sequences from {len(train_videos)} videos")
    print(f"Val: {len(val_indices)} sequences from {len(val_videos)} videos")
    print(f"Test: {len(test_indices)} sequences from {len(test_videos)} videos")

    # Create datasets
    train_dataset = BehaviorDataset(
        [sequences[i] for i in train_indices],
        [labels[i] for i in train_indices]
    )
    val_dataset = BehaviorDataset(
        [sequences[i] for i in val_indices],
        [labels[i] for i in val_indices]
    )
    test_dataset = BehaviorDataset(
        [sequences[i] for i in test_indices],
        [labels[i] for i in test_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True if config.NUM_WORKERS > 0 else False
    )

    # ==========================================================================
    # Model Initialization
    # ==========================================================================
    print("\n[3/5] Initializing model...")

    model = BehaviorClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=config.NUM_CHANNELS,
        kernel_size=config.KERNEL_SIZE,
        dropout=config.DROPOUT
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ==========================================================================
    # Training
    # ==========================================================================
    print("\n[4/5] Training model...")

    trainer = Trainer(model, config, processor.class_weights)
    history = trainer.train(train_loader, val_loader, config.NUM_EPOCHS)

    # Save model
    model_path = config.OUTPUT_DIR / 'tcn_behavior_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'num_channels': config.NUM_CHANNELS,
            'kernel_size': config.KERNEL_SIZE,
            'dropout': config.DROPOUT
        },
        'action_to_idx': processor.action_to_idx,
        'idx_to_action': processor.idx_to_action
    }, model_path)
    print(f"Saved model to {model_path}")

    # ==========================================================================
    # Evaluation
    # ==========================================================================
    print("\n[5/5] Evaluating model...")

    test_loss, test_f1, test_preds, test_labels = trainer.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    # Visualization
    visualizer = Visualizer(config.OUTPUT_DIR, processor.idx_to_action)
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(test_labels, test_preds)
    visualizer.plot_per_class_metrics(test_labels, test_preds)
    report = visualizer.save_classification_report(test_labels, test_preds)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    print("\nClassification Report:")
    print(report)

    return model, processor, history


if __name__ == "__main__":
    main()
