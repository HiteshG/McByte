"""
Colab Inference Entry Point for McByte.

Unified interface for running multi-object tracking with:
- YOLOv8 detection (custom weights)
- Modular tracker selection
- SAM + Cutie mask propagation
- MP4 video input/output
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import cv2
import numpy as np
import torch

# Add parent directory to path for imports
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from tools.yolov8_predictor import YOLOv8Predictor
from tools.tracker_interface import (
    create_tracker,
    list_available_trackers,
    Track,
    TrackerOutput,
    McByteTrackerAdapter,
)
from tools.demo_track import filter_detections_by_class
from yolox.tracker.class_config import (
    TRACK_CLASSES as DEFAULT_TRACK_CLASSES,
    SPECIAL_CLASSES as DEFAULT_SPECIAL_CLASSES,
    CLASS_NAMES as DEFAULT_CLASS_NAMES,
)


def run_inference(
    video_path: str,
    yolov8_weights: str,
    output_path: Optional[str] = None,
    # Tracker options
    tracker_type: str = "mcbyte",
    track_thresh: float = 0.6,
    track_buffer: int = 30,
    # Mask options
    sam_checkpoint: Optional[str] = None,
    sam_type: str = "vit_b",
    cutie_weights: Optional[str] = None,
    enable_masks: bool = True,
    # Class filtering options
    track_classes: Optional[List[int]] = None,
    special_classes: Optional[List[int]] = None,
    no_class_filter: bool = False,
    # Other options
    device: str = "auto",
    class_names: Optional[List[str]] = None,
    classes: Optional[List[int]] = None,
    fps: Optional[int] = None,
    vis_type: str = "basic",
    conf_thresh: float = 0.01,
    start_frame: int = 1,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    Run multi-object tracking with mask propagation on a video.

    Args:
        video_path: Path to input video file
        yolov8_weights: Path to YOLOv8 weights (.pt file)
        output_path: Path for output video. If None, auto-generated.

        # Tracker options
        tracker_type: Tracker to use. Options: "mcbyte", "bytetrack", "sort",
                     "deepsort", "botsort", "ocsort"
        track_thresh: Detection confidence threshold for tracking
        track_buffer: Frames to keep lost tracks before removal

        # Mask options
        sam_checkpoint: Path to SAM weights. Required for SAM1, ignored for SAM2+.
        sam_type: SAM model type. Options:
                 - SAM1: "vit_b", "vit_l", "vit_h" (requires sam_checkpoint)
                 - SAM2+: HuggingFace model ID (e.g., "facebook/sam2.1-hiera-large")
        cutie_weights: Path to Cutie weights. If None, uses default.
        enable_masks: Whether to enable mask propagation

        # Class filtering options
        track_classes: List of class IDs to track. If None, uses defaults from class_config.
        special_classes: List of class IDs where only max-confidence detection is kept
                        (useful for small objects like puck). If None, uses defaults.
        no_class_filter: If True, disables class filtering (track all detected classes)

        # Other options
        device: Device to use ('cuda', 'cpu', 'auto')
        class_names: Optional list of class names for your model
        classes: Optional list of class indices to detect (None = all)
        fps: Output video FPS. If None, uses input video FPS.
        vis_type: Visualization type: "basic" or "no_vis"
        conf_thresh: Confidence threshold for detections
        start_frame: Frame to start processing from (1-indexed)
        max_frames: Maximum frames to process (None = all)
        verbose: Print progress information

    Returns:
        Path to the output video file
    """
    # Handle device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[McByte] Using device: {device}")
        print(f"[McByte] Tracker: {tracker_type}")
        print(f"[McByte] Masks enabled: {enable_masks}")

    # Determine class filtering settings
    if no_class_filter:
        _track_classes = None
        _special_classes = None
        if verbose:
            print("[McByte] Class filtering: DISABLED (tracking all classes)")
    else:
        _track_classes = track_classes if track_classes is not None else DEFAULT_TRACK_CLASSES
        _special_classes = special_classes if special_classes is not None else DEFAULT_SPECIAL_CLASSES
        if verbose:
            print(f"[McByte] Track classes: {_track_classes}")
            print(f"[McByte] Special classes (max-conf only): {_special_classes}")

    # Create output directory
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path("YOLOX_outputs") / "colab_inference" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path if not provided
    if output_path is None:
        input_name = Path(video_path).stem
        output_path = str(output_dir / f"{input_name}_tracked.mp4")

    # Initialize YOLOv8 predictor
    if verbose:
        print(f"[McByte] Loading YOLOv8 model: {yolov8_weights}")

    predictor = YOLOv8Predictor(
        model_path=yolov8_weights,
        device=device,
        conf_thresh=conf_thresh,
        class_names=class_names,
        classes=classes,
    )

    # Initialize tracker
    if verbose:
        print(f"[McByte] Initializing {tracker_type} tracker...")

    tracker = create_tracker(
        tracker_type=tracker_type,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        frame_rate=fps if fps else 30,
        save_folder=str(output_dir),
    )

    # Initialize mask manager if enabled
    mask_manager = None
    if enable_masks:
        if verbose:
            print("[McByte] Initializing SAM + Cutie for mask propagation...")

        from mask_propagation.mask_manager import MaskManager
        mask_manager = MaskManager(
            sam_checkpoint=sam_checkpoint,
            sam_type=sam_type,
            cutie_weights=cutie_weights,
            device=device,
        )

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = input_fps

    if verbose:
        print(f"[McByte] Input video: {width}x{height} @ {input_fps:.1f} FPS, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Import visualization function
    from yolox.utils.visualize import plot_tracking_basic

    # Processing state
    prediction = None
    tracklet_mask_dict = None
    mask_avg_prob_dict = None
    prediction_colors_preserved = None
    img_info_prev = None
    online_tlwhs = []
    online_ids = []
    new_tracks = []
    removed_tracks_ids = []
    results = []

    # Process video
    frame_idx = 0
    processed_frames = 0

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            while True:
                ret, frame = cap.read()
                frame_idx += 1

                if not ret:
                    break

                # Skip frames before start_frame
                if frame_idx < start_frame:
                    continue

                # Check max_frames limit
                if max_frames is not None and processed_frames >= max_frames:
                    break

                processed_frames += 1
                frame_id = processed_frames

                if verbose and frame_id % 10 == 0:
                    print(f"[McByte] Processing frame {frame_id}/{total_frames - start_frame + 1}")

                # Run detection
                outputs, img_info = predictor.inference(frame)

                # Apply class filtering and special class handling
                if outputs[0] is not None and len(outputs[0]) > 0:
                    outputs = (filter_detections_by_class(
                        outputs[0],
                        track_classes=_track_classes,
                        special_classes=_special_classes
                    ),)

                if outputs[0] is not None and len(outputs[0]) > 0:
                    # Update masks (if enabled and not first frame)
                    if enable_masks and mask_manager is not None and frame_id > 1:
                        (
                            prediction,
                            tracklet_mask_dict,
                            mask_avg_prob_dict,
                            prediction_colors_preserved,
                        ) = mask_manager.get_updated_masks(
                            img_info,
                            img_info_prev,
                            frame_id,
                            online_tlwhs,
                            online_ids,
                            new_tracks,
                            removed_tracks_ids,
                        )

                    # Update tracker
                    if isinstance(tracker, McByteTrackerAdapter):
                        # McByte tracker with mask support
                        tracker_output = tracker.update(
                            outputs[0],
                            img_info,
                            test_size=predictor.test_size,
                            prediction_mask=prediction,
                            tracklet_mask_dict=tracklet_mask_dict if tracklet_mask_dict else {},
                            mask_avg_prob_dict=mask_avg_prob_dict if mask_avg_prob_dict else {},
                            vis_type=vis_type,
                        )
                    else:
                        # Other trackers
                        tracker_output = tracker.update(outputs[0], img_info)

                    # Extract tracking results
                    online_targets = tracker_output.online_targets
                    removed_tracks_ids = tracker_output.removed_track_ids
                    new_tracks = tracker_output.new_tracks

                    # Convert Track objects to tlwh lists for mask manager
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []

                    for t in online_targets:
                        tlwh = t.last_det_tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        # Save results in MOT format with class_id in last column
                        cls_id = t.class_id if hasattr(t, 'class_id') and t.class_id is not None else -1
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                            f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,{cls_id}\n"
                        )

                    # Visualization
                    if vis_type == "basic":
                        online_im, _, _ = plot_tracking_basic(
                            img_info['raw_img'],
                            online_tlwhs,
                            online_ids,
                            frame_id=frame_id,
                            prediction_mask=prediction_colors_preserved,
                        )
                    else:
                        online_im = img_info['raw_img']
                else:
                    online_im = frame
                    online_tlwhs = []
                    online_ids = []
                    new_tracks = []
                    removed_tracks_ids = []

                # Store previous frame info for mask propagation
                img_info_prev = img_info

                # Write frame to output
                vid_writer.write(online_im)

    # Cleanup
    cap.release()
    vid_writer.release()

    # Save MOT format results
    results_path = output_dir / f"{Path(video_path).stem}_results.txt"
    with open(results_path, 'w') as f:
        f.writelines(results)

    if verbose:
        print(f"[McByte] Processing complete!")
        print(f"[McByte] Output video: {output_path}")
        print(f"[McByte] MOT results: {results_path}")

    return output_path


def validate_setup(
    yolov8_weights: str,
    sam_checkpoint: Optional[str] = None,
    sam_type: str = "vit_b",
    cutie_weights: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate that all components are properly configured.

    Args:
        yolov8_weights: Path to YOLOv8 weights
        sam_checkpoint: Path to SAM checkpoint
        sam_type: SAM model type
        cutie_weights: Path to Cutie weights
        class_names: Optional custom class names
        device: Device to use
        verbose: Print validation progress

    Returns:
        Dict with validation results
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "device": device,
        "yolov8": False,
        "sam": False,
        "cutie": False,
        "errors": [],
    }

    # Check device
    if verbose:
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
        print(f"[1/4] Device: {device} ({gpu_name if device == 'cuda' else 'CPU'})")

    # Check YOLOv8
    if verbose:
        print("[2/4] Checking YOLOv8...")
    try:
        from ultralytics import YOLO
        model = YOLO(yolov8_weights)
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = model(test_img, verbose=False)
        results["yolov8"] = True
        results["yolov8_classes"] = len(model.names)
        if verbose:
            print(f"  [OK] YOLOv8 loaded: {len(model.names)} classes")
            if class_names:
                print(f"  [OK] Custom classes: {class_names}")
        del model
    except Exception as e:
        results["errors"].append(f"YOLOv8: {str(e)}")
        if verbose:
            print(f"  [FAIL] YOLOv8: {str(e)}")

    # Check SAM
    if verbose:
        print("[3/4] Checking SAM...")
    try:
        # Check if sam_type is a HuggingFace model ID (contains "/")
        is_huggingface_model = "/" in sam_type

        if is_huggingface_model:
            # SAM2+ via HuggingFace
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor.from_pretrained(sam_type)
            predictor.model.to(device)
            sam = predictor.model
            if verbose:
                print(f"  [OK] SAM2 loaded from HuggingFace: {sam_type}")
        else:
            # SAM1 via local checkpoint
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
            sam.to(device)
            predictor = SamPredictor(sam)

        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        predictor.set_image(test_img)
        masks, scores, _ = predictor.predict(
            box=np.array([100, 100, 300, 300]),
            multimask_output=False,
        )
        results["sam"] = True
        if verbose:
            print(f"  [OK] SAM working: mask shape {masks.shape}, score {scores[0]:.3f}")
        del sam, predictor
    except Exception as e:
        results["errors"].append(f"SAM: {str(e)}")
        if verbose:
            print(f"  [FAIL] SAM: {str(e)}")

    # Check Cutie
    if verbose:
        print("[4/4] Checking Cutie...")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "mask_propagation" / "Cutie"))
        from omegaconf import open_dict
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        from cutie.model.cutie import CUTIE
        from cutie.inference.utils.args_utils import get_dataset_cfg

        GlobalHydra.instance().clear()
        initialize(version_base='1.3.2', config_path="../mask_propagation/Cutie/cutie/config")
        cfg = compose(config_name="eval_config")

        if cutie_weights is None:
            cutie_weights = str(
                Path(__file__).parent.parent
                / "mask_propagation"
                / "Cutie"
                / "weights"
                / "cutie-base-mega.pth"
            )

        with open_dict(cfg):
            cfg['weights'] = cutie_weights

        _ = get_dataset_cfg(cfg)

        cutie = CUTIE(cfg).to(device).eval()
        weights = torch.load(cutie_weights, map_location=device)
        cutie.load_weights(weights)
        results["cutie"] = True
        if verbose:
            print("  [OK] Cutie loaded successfully")
        del cutie
    except Exception as e:
        results["errors"].append(f"Cutie: {str(e)}")
        if verbose:
            print(f"  [FAIL] Cutie: {str(e)}")

    # Cleanup
    torch.cuda.empty_cache()

    # Summary
    all_passed = results["yolov8"] and results["sam"] and results["cutie"]
    results["all_passed"] = all_passed

    if verbose:
        if all_passed:
            print("\n[OK] All validation checks passed!")
        else:
            print(f"\n[FAIL] Some checks failed: {results['errors']}")

    return results


def process_video_frames(
    input_path: str,
    output_dir: str,
    start_frame: int = 1,
    max_frames: Optional[int] = None,
) -> List[str]:
    """
    Extract frames from a video to a directory.

    Args:
        input_path: Path to input video
        output_dir: Directory to save frames
        start_frame: First frame to extract (1-indexed)
        max_frames: Maximum frames to extract

    Returns:
        List of frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    frame_paths = []
    frame_idx = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        frame_idx += 1

        if not ret:
            break

        if frame_idx < start_frame:
            continue

        if max_frames is not None and extracted >= max_frames:
            break

        extracted += 1
        frame_path = os.path.join(output_dir, f"{extracted:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: float = 30.0,
    pattern: str = "*.jpg",
) -> str:
    """
    Combine frames into a video.

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        fps: Frames per second
        pattern: Glob pattern for frame files

    Returns:
        Path to output video
    """
    from glob import glob

    frame_paths = sorted(glob(os.path.join(frame_dir, pattern)))
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir} matching {pattern}")

    # Get frame dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        vid_writer.write(frame)

    vid_writer.release()
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="McByte Colab Inference")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", required=True, help="YOLOv8 weights path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument(
        "--tracker",
        default="mcbyte",
        choices=list_available_trackers(),
        help="Tracker type",
    )
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-type", default="vit_b")
    parser.add_argument("--cutie-weights", default=None)
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vis-type", default="basic", choices=["basic", "no_vis"])
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    # Class filtering arguments
    parser.add_argument(
        "--track-classes",
        type=int,
        nargs="+",
        default=None,
        help="Class IDs to track (e.g., --track-classes 3 4 5 6)"
    )
    parser.add_argument(
        "--special-classes",
        type=int,
        nargs="+",
        default=None,
        help="Classes where only max-confidence detection is kept (e.g., --special-classes 5)"
    )
    parser.add_argument(
        "--no-class-filter",
        action="store_true",
        help="Disable class filtering (track all detected classes)"
    )

    args = parser.parse_args()

    if args.validate:
        validate_setup(
            yolov8_weights=args.weights,
            sam_checkpoint=args.sam_checkpoint,
            sam_type=args.sam_type,
            cutie_weights=args.cutie_weights,
            device=args.device,
        )
    else:
        output = run_inference(
            video_path=args.video,
            yolov8_weights=args.weights,
            output_path=args.output,
            tracker_type=args.tracker,
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            sam_checkpoint=args.sam_checkpoint,
            sam_type=args.sam_type,
            cutie_weights=args.cutie_weights,
            enable_masks=not args.no_masks,
            track_classes=args.track_classes,
            special_classes=args.special_classes,
            no_class_filter=args.no_class_filter,
            device=args.device,
            vis_type=args.vis_type,
        )
        print(f"Output saved to: {output}")
