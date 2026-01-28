"""
YOLOv8 Predictor wrapper for McByte tracking pipeline.

Adapts YOLOv8 output to tracker-compatible format [x1, y1, x2, y2, score].
"""

import cv2
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any


class YOLOv8Predictor:
    """
    YOLOv8 wrapper that provides a consistent interface for the tracking pipeline.

    Outputs detections in format compatible with McByte tracker:
    - outputs: tensor of shape [N, 5] with [x1, y1, x2, y2, score]
    - img_info: dict with frame metadata
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf_thresh: float = 0.01,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        classes: Optional[List[int]] = None,
        img_size: int = 640,
    ):
        """
        Initialize YOLOv8 predictor.

        Args:
            model_path: Path to YOLOv8 weights (.pt file)
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
            class_names: Optional list of class names (for custom models)
            classes: Optional list of class indices to detect (None = all classes)
            img_size: Input image size for inference
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )

        # Handle device selection
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.img_size = img_size

        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)

        # Set class names
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = self.model.names

        self.num_classes = len(self.class_names)

        # For compatibility with existing pipeline
        self.test_size = (img_size, img_size)

    def inference(self, img: Any) -> Tuple[List[Optional[np.ndarray]], Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            img: Image as numpy array (BGR) or path to image file

        Returns:
            outputs: List containing detection tensor [N, 5] with [x1, y1, x2, y2, score]
                    or [None] if no detections
            img_info: Dict with keys:
                - 'id': frame id (always 0)
                - 'file_name': filename if path provided, else None
                - 'height': image height
                - 'width': image width
                - 'raw_img': original image as numpy array (BGR)
        """
        img_info = {"id": 0}

        # Handle image input
        if isinstance(img, str):
            img_info["file_name"] = img.split("/")[-1] if "/" in img else img
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"Could not read image: {img_info['file_name']}")
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # Run YOLOv8 inference
        results = self.model(
            img,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            imgsz=self.img_size,
            verbose=False,
            device=self.device,
        )

        # Extract detections
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # Get bounding boxes in xyxy format
            xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]

            # Get confidence scores
            conf = boxes.conf.cpu().numpy()  # [N]

            # Combine into [N, 5] array: [x1, y1, x2, y2, score]
            detections = np.zeros((len(xyxy), 5), dtype=np.float32)
            detections[:, :4] = xyxy
            detections[:, 4] = conf

            # Return as list (for compatibility with existing pipeline)
            outputs = [detections]
        else:
            outputs = [None]

        return outputs, img_info

    def warmup(self, img_size: Optional[Tuple[int, int]] = None):
        """
        Warmup the model with a dummy inference.

        Args:
            img_size: Optional (height, width) tuple, defaults to self.img_size
        """
        if img_size is None:
            img_size = (self.img_size, self.img_size)

        dummy_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        _ = self.inference(dummy_img)

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_id, f"class_{class_id}")
        elif isinstance(self.class_names, list):
            if 0 <= class_id < len(self.class_names):
                return self.class_names[class_id]
        return f"class_{class_id}"


def create_yolov8_predictor(
    model_path: str,
    device: str = 'auto',
    conf_thresh: float = 0.01,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> YOLOv8Predictor:
    """
    Factory function to create YOLOv8Predictor.

    Args:
        model_path: Path to YOLOv8 weights
        device: Device ('cuda', 'cpu', 'auto')
        conf_thresh: Confidence threshold
        class_names: Optional custom class names
        **kwargs: Additional arguments passed to YOLOv8Predictor

    Returns:
        Configured YOLOv8Predictor instance
    """
    return YOLOv8Predictor(
        model_path=model_path,
        device=device,
        conf_thresh=conf_thresh,
        class_names=class_names,
        **kwargs
    )
