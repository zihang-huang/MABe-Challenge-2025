
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from tcn_behavior_model import Config, DataProcessor, BehaviorClassifier, FocalLoss, BehaviorDataset

def test_model_components():
    print("Testing model components...")

    # 1. Test Config
    config = Config()
    print("Config loaded.")

    # 2. Test Focal Loss
    criterion = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)
    logits = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss = criterion(logits, targets)
    print(f"Focal Loss forward pass successful. Loss: {loss.item()}")

    # 3. Test Model Architecture
    input_dim = 32
    num_classes = 5
    model = BehaviorClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        num_channels=config.NUM_CHANNELS,
        kernel_size=config.KERNEL_SIZE,
        dropout=config.DROPOUT
    )
    
    dummy_input = torch.randn(4, 64, input_dim) # [B, T, F]
    output = model(dummy_input)
    print(f"Model forward pass successful. Output shape: {output.shape}")
    assert output.shape == (4, 64, num_classes)

    # 4. Test DataProcessor (Mocking)
    processor = DataProcessor(Path("dummy_path"), config)
    
    # Mock extract_agent_target_features to test new features logic
    # We need to construct a small dataframe to test feature extraction
    print("Testing feature extraction logic...")
    
    # Create dummy features df
    frames = list(range(10))
    feature_cols = config.FEATURE_COLS
    
    data = {
        'video_frame': frames * 2,
        'mouse_id': [1]*10 + [2]*10,
    }
    for col in feature_cols:
        data[col] = np.random.rand(20)
        
    features_df = pd.DataFrame(data)
    
    # Test extraction
    # We need to make sure the indices line up
    combined = processor.extract_agent_target_features(features_df, 1, 2)
    
    if combined is not None:
        # Expected dim: 
        # Agent (14) + Target (14) + RelPos (2) + Dist (1) + RelHead (1) + RelVel (2) + RelSpeed (1)
        # Total = 14 + 14 + 2 + 1 + 1 + 2 + 1 = 35
        print(f"Feature extraction successful. Shape: {combined.shape}")
        # Note: Input dim in real run will depend on this.
        assert combined.shape[1] == 35, f"Expected 35 features, got {combined.shape[1]}"
    else:
        print("Feature extraction returned None (unexpected for dummy data).")

    print("All component tests passed!")

if __name__ == "__main__":
    test_model_components()
