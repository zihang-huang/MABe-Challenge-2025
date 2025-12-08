# MABe Challenge 2025: Data Preprocessing Plan

## 1. Dataset Overview

This project involves multi-agent temporal behavior detection from mouse tracking data. The dataset consists of:

- **Metadata** (`train.csv`): Video/experiment-level information including mouse attributes, arena specifications, and recording parameters
- **Tracking Data** (`train_tracking/`): Frame-by-frame (x, y) coordinates for multiple body parts per mouse
- **Annotations** (`train_annotation/`): Behavior labels with agent-target pairs and temporal boundaries (start/stop frames)

### Key Characteristics
- **Multi-agent**: 2-4 mice per video (variable number of agents)
- **Multi-body-part tracking**: 5-18 body parts depending on the lab/setup
- **Pairwise behaviors**: Labels encode agent-target-action triplets (e.g., mouse1 → mouse2: chase)
- **Heterogeneous sources**: Data from multiple labs with varying setups (different FPS, arena shapes, resolutions)

---

## 2. Metadata Preprocessing (`train.csv`)

### 2.1 Columns to Remove

| Column(s) | Rationale |
|-----------|-----------|
| `lab_id` | Identifier only; may introduce unwanted bias if used as feature |
| `video_id` | Unique identifier; used for file matching only, not modeling |
| `tracking_method` | Uniform (DeepLabCut/MARS); no predictive value |
| `mouse{1-4}_id` | Arbitrary identifiers within labs |

### 2.2 Columns Requiring Careful Consideration

| Column(s) | Decision | Rationale |
|-----------|----------|-----------|
| `mouse{1-4}_strain`, `color`, `sex`, `age`, `condition` | **Defer removal** | May correlate with behavior patterns (e.g., strain-specific aggression). Evaluate via feature importance analysis before discarding. |
| `video_duration_sec` | **Remove for model input** | Tracking data already encodes temporal extent; avoid information leakage |
| `body_parts_tracked` | **Use for validation** | Critical for alignment with tracking files; verify consistency |
| `behaviors_labeled` | **Use for validation** | Ground truth reference; extract unique behavior vocabulary |

### 2.3 Columns to Retain for Normalization

| Column | Purpose |
|--------|---------|
| `frames_per_second` | Temporal normalization (standardize to common FPS) |
| `pix_per_cm_approx` | Spatial normalization (convert pixels → centimeters) |
| `video_width_pix`, `video_height_pix` | Coordinate normalization to [0, 1] range |
| `arena_width_cm`, `arena_height_cm` | Alternative spatial reference frame |
| `arena_shape` | Context feature (square vs. rectangular affects movement patterns) |
| `arena_type` | Context feature (familiar vs. resident-intruder paradigms) |

---

## 3. Tracking Data Preprocessing (`train_tracking/`)

### 3.1 Spatial Normalization Pipeline

```
Raw Coordinates (pixels)
        │
        ▼
┌─────────────────────────────────┐
│  1. Pixels → Centimeters        │
│     x_cm = x_pix / pix_per_cm   │
│     y_cm = y_pix / pix_per_cm   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  2. Arena-Relative Coordinates  │
│     x_norm = x_cm / arena_width │
│     y_norm = y_cm / arena_height│
│     Range: [0, 1]               │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  3. Center at Arena Midpoint    │
│     x_centered = x_norm - 0.5   │
│     y_centered = y_norm - 0.5   │
│     Range: [-0.5, 0.5]          │
└─────────────────────────────────┘
```

### 3.2 Temporal Normalization

**Challenge**: Videos have varying FPS (25 Hz or 30 Hz)

**Strategy**:
1. Resample all tracking data to a uniform frame rate (e.g., 30 FPS)
2. Use linear interpolation for upsampling (25 → 30 FPS)
3. Adjust annotation frame indices accordingly: `new_frame = old_frame * (target_fps / source_fps)`

### 3.3 Body Part Handling

**Challenge**: Different labs track different body part sets (5 to 18 keypoints)

**Strategy Options**:
| Approach | Description | Trade-offs |
|----------|-------------|------------|
| **Common Subset** | Use only shared body parts across all videos | Loss of information; limited to ~5 parts (body_center, ear_left, ear_right, nose, tail_base) |
| **Padding/Masking** | Pad missing body parts with NaN + attention mask | Preserves all data; requires model architecture support |
| **Lab-Specific Models** | Train separate encoders per tracking configuration | Higher complexity; may improve performance |

**Recommended**: Start with common subset for baseline, then explore padding/masking.

### 3.4 Missing Value Handling

1. **Identify occlusions**: Frames where body parts have low confidence or are absent
2. **Interpolation**: Short gaps (< 5 frames) → linear interpolation
3. **Masking**: Longer gaps → mask tokens for transformer-based models
4. **Validation**: Flag videos with excessive missing data (> 10% of frames)

---

## 4. Feature Engineering

> These are suggestions from Claude Opus 4.5, we don't need to accept all the decision from it.

> I just ask it to provide some ideas that we can start with. 

### 4.1 Derived Spatial Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Inter-mouse distance | `dist(mouse_i.body_center, mouse_j.body_center)` | Proximity is key indicator for social behaviors |
| Relative heading | `atan2(dy, dx)` between noses | Orientation predicts approach vs. avoid |
| Body elongation | `dist(nose, tail_base)` | Posture indicator (stretched vs. crouched) |
| Velocity | `(x_t - x_{t-1}) / dt` | Speed differentiates chase from idle proximity |
| Acceleration | `(v_t - v_{t-1}) / dt` | Sudden changes indicate behavior transitions |

### 4.2 Temporal Features

| Feature | Description |
|---------|-------------|
| Sliding window statistics | Mean, std, min, max over N-frame windows |
| Relative motion | Movement direction relative to other mice |
| Social attention | Whether nose points toward another mouse |

### 4.3 Graph-Based Representation

For multi-agent modeling, represent each frame as a graph:
- **Nodes**: Individual mice (with body part features)
- **Edges**: Pairwise relationships (distance, relative velocity, attention)

---

## 5. Annotation Preprocessing (`train_annotation/`)

### 5.1 Label Structure

Current format: `(agent_id, target_id, action, start_frame, stop_frame)`

### 5.2 Conversion to Frame-Level Labels

For sequence modeling, convert to per-frame multi-label format:

```
Frame 0: []
Frame 1: []
Frame 2: [(1, 3, chase)]
Frame 3: [(1, 3, chase)]
...
Frame 54: [(1, 3, chase)]
Frame 55: []
```

### 5.3 Handling Overlapping Behaviors

- Multiple behaviors can occur simultaneously (e.g., mouse1→mouse2: chase AND mouse3→mouse4: avoid)
- Use multi-hot encoding per (agent, target) pair

### 5.4 Behavior Vocabulary

Extract and standardize behavior labels across labs:
- `approach`, `attack`, `avoid`, `chase`, `chaseattack`, `submit`
- `disengage`, `mount`, `sniff`, `sniffgenital`
- `rear`, `selfgroom`, `shepherd`

---

## 6. Data Quality Checks

| Check | Action |
|-------|--------|
| Coordinate bounds | Verify all (x, y) within video dimensions |
| Frame count consistency | Match tracking frames to video_duration × FPS |
| Annotation frame bounds | Ensure start_frame < stop_frame ≤ max_frame |
| Mouse count alignment | Verify mice in annotations exist in tracking data |
| Duplicate detection | Identify and remove duplicate rows |

---

## 7. Output Format

### 7.1 Preprocessed Tracking Files
```
Format: Parquet or HDF5
Schema:
  - video_id: string
  - frame: int
  - mouse_id: int
  - features: float32[N]  # Normalized coordinates + derived features
```

### 7.2 Preprocessed Annotation Files
```
Format: Parquet
Schema:
  - video_id: string
  - frame: int
  - agent_id: int
  - target_id: int
  - behavior: categorical
```

### 7.3 Metadata Lookup Table
```
Format: JSON or Parquet
Contents: Arena parameters, FPS, normalization constants per video
```

---

## 8. Implementation Tasks

### Core Preprocessing
- [ ] Load and parse `train.csv` metadata
- [ ] Implement spatial normalization (pixels → normalized arena coordinates)
- [ ] Implement temporal resampling (standardize FPS)
- [ ] Handle missing body parts (common subset extraction)
- [ ] Convert annotations to frame-level format

### Feature Engineering & Validation
- [ ] Compute derived features (velocity, acceleration, inter-mouse distance)
- [ ] Run data quality checks
- [ ] Generate train/validation splits (stratified by lab to test generalization)
- [ ] Export preprocessed data to efficient storage format