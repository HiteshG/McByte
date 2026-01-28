# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

McByte is a CVPRW 2025 multi-object tracking (MOT) framework that combines detection, tracking, and temporal mask propagation. It requires **no training** - it uses pre-trained models for all components.

**Core Components:**
- **YOLOX**: Object detection (pre-trained weights)
- **ByteTrack-style tracker**: Kalman filtering + Hungarian matching
- **Cutie**: Temporal mask propagation
- **SAM**: Initial frame segmentation

## Commands

### Running the Demo

```bash
# Frame folder input (default)
python tools/demo_track.py --path /path/to/frames

# Video file input
python tools/demo_track.py --path video.mp4 --demo video

# With custom detector checkpoint
python tools/demo_track.py --path input --ckpt pretrained/bytetrack_x_mot17.pth.tar

# Skip detection (use pre-computed detections)
python tools/demo_track.py --path input --det_path detections.txt

# Key arguments
--vis_type full|basic|no_vis    # Visualization level
--start_frame_no N              # Start from frame N
--track_thresh 0.5              # Detection confidence threshold
--track_buffer 30               # Frames to keep lost tracks
--cmc-method orb|ecc|files      # Camera motion compensation
```

### Build/Setup

```bash
# Compile C++ extensions (required after code changes)
python3 setup.py develop

# Fix numpy version issues
pip install --upgrade numpy==1.23
```

### Output Location

Results are saved to: `YOLOX_outputs/yolox_x_mix_det/track_vis/{timestamp}/`
- Tracked frames with boxes/masks
- `logging_info.txt` - per-frame associations, costs, masks
- `{timestamp}.txt` - MOT format output

## Architecture

### Entry Point & Flow

**Main entry**: `tools/demo_track.py`

```
Input (Video/Frames)
    ↓
YOLOX Detection → bboxes + scores
    ↓
McByteTracker.update() (with mask guidance)
    ├── Kalman prediction
    ├── Camera motion compensation (GMC)
    ├── First association (high-conf detections, mask-enhanced)
    ├── Second association (low-conf detections)
    └── Track activation/management
    ↓
MaskManager.get_updated_masks()
    ├── Frame 1: SAM segmentation + Cutie initialization
    ├── Frame 2+: Cutie temporal propagation
    └── Add/remove masks as tracks change
    ↓
Output (frames + MOT text file + logs)
```

### Key Files

| File | Purpose |
|------|---------|
| `tools/demo_track.py` | Main entry point, orchestrates detection→tracking→masks |
| `yolox/tracker/mcbyte_tracker.py` | Core tracker: `STrack` class, `McByteTracker.update()` |
| `yolox/tracker/matching.py` | Association metrics: IoU, Hungarian matching |
| `yolox/tracker/kalman_filter.py` | 8D state Kalman filter for motion |
| `mask_propagation/mask_manager.py` | SAM + Cutie integration, mask lifecycle |
| `exps/example/mot/yolox_x_mix_det.py` | Default detector config (YOLOX-X) |

### Tracking Algorithm (mcbyte_tracker.py)

The `McByteTracker.update()` method (lines 387+) implements:

1. **Detection extraction**: Split into high-conf (>threshold) and low-conf (0.1-threshold)
2. **Prediction**: Kalman filter predicts track positions, GMC compensates camera motion
3. **First association**: High-conf detections matched to tracks using IoU + **mask overlap metrics**
4. **Second association**: Low-conf detections for unconfirmed/lost tracks
5. **Track management**: Activate new, mark lost, remove stale

**Mask-enhanced matching** (in `conditioned_assignment`):
- mm1: mask pixels in bbox / total mask pixels (threshold: 0.9)
- mm2: mask pixels in bbox / total bbox area (threshold: 0.05)
- If mask confident (>0.6) and metrics pass, cost is reduced

### Critical Constants (mcbyte_tracker.py:14-21)

```python
MIN_MASK_AVG_CONF = 0.6           # Minimum mask confidence to use
MIN_MM1 = 0.9                      # Mask coverage threshold
MIN_MM2 = 0.05                     # Mask fill threshold
MAX_COST_1ST_ASSOC_STEP = 0.9      # First association cost threshold
MAX_COST_2ND_ASSOC_STEP = 0.5      # Second association cost threshold
```

### Mask Manager (mask_manager.py)

Handles three operations:
- **Initialize**: First frame - SAM segments detected objects, Cutie receives initial masks
- **Propagate**: Subsequent frames - Cutie propagates masks temporally
- **Lifecycle**: Add masks for new tracks, remove masks for lost tracks

Key constant: `MASK_CREATION_BB_OVERLAP_THRESHOLD = 0.6` - delays mask creation for overlapping objects

### Coordinate Formats

- **tlwh**: (top-left-x, top-left-y, width, height) - internal storage
- **tlbr**: (top-left-x, top-left-y, bottom-right-x, bottom-right-y) - matching
- **xywh**: (center-x, center-y, width, height) - Kalman filter state

## Model Weights

| Component | File | Location |
|-----------|------|----------|
| YOLOX (SportsMOT) | `yolox_x_sports_mix.pth.tar` | `pretrained/` |
| Cutie | `cutie-base-mega.pth` | `mask_propagation/Cutie/weights/` |
| SAM | `sam_vit_b_01ec64.pth` | `sam_models/` |

Download links in INSTALLATION.md. Backup: [Google Drive](https://drive.google.com/drive/folders/1yzzJk9dpJUY3lIHdkkFyGtKL2F-FenN6)

## Memory Optimization

Edit `mask_propagation/Cutie/cutie/config/eval_config.yaml`:
```yaml
max_internal_size: -1   # -1 = match input resolution
                        # Set to 540 for 1080p input (50% = 4x memory reduction)
```

## Code Style

From `setup.cfg`:
- Line length: 100
- isort: torch, torchvision, timm, thop as deeplearning imports
