"""
Modular Tracker Interface for McByte.

Provides an abstract base class and concrete implementations for different
tracking algorithms, allowing easy swapping of trackers while maintaining
compatibility with the mask propagation pipeline.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Any
import numpy as np
import torch


# Named tuple for tracker output
TrackerOutput = namedtuple('TrackerOutput', [
    'online_targets',      # List[Track] - active tracks this frame
    'removed_track_ids',   # List[int] - IDs of removed tracks
    'new_tracks'           # List[Track] - newly created/confirmed tracks
])


@dataclass
class Track:
    """
    Minimal track interface for mask pipeline compatibility.

    All trackers must output Track objects with these attributes.
    """
    track_id: int
    tlwh: np.ndarray       # [top_left_x, top_left_y, width, height]
    last_det_tlwh: np.ndarray  # Last detection bbox (for mask creation)
    score: float

    @property
    def tlbr(self) -> np.ndarray:
        """Convert to [top_left_x, top_left_y, bottom_right_x, bottom_right_y]."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class BaseTrackerInterface(ABC):
    """
    Abstract base class for tracker implementations.

    Any tracker must implement the `update` method to work with the mask pipeline.
    """

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any]
    ) -> TrackerOutput:
        """
        Update tracker with new detections.

        Args:
            detections: [N, 5] array with [x1, y1, x2, y2, score]
            img_info: dict with 'height', 'width', 'raw_img'

        Returns:
            TrackerOutput with:
                - online_targets: list of Track objects for active tracks
                - removed_track_ids: list of removed track IDs
                - new_tracks: list of newly created Track objects
        """
        raise NotImplementedError

    def reset(self):
        """Reset tracker state for a new video."""
        pass


class McByteTrackerAdapter(BaseTrackerInterface):
    """
    Adapter for the original McByteTracker (mask-enhanced matching).

    This is the default tracker that uses mask information to improve association.
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        track_buffer: int = 30,
        frame_rate: int = 30,
        save_folder: str = ".",
        cmc_method: str = "orb",
    ):
        from yolox.tracker.mcbyte_tracker import McByteTracker

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.track_thresh = track_thresh
        args.track_buffer = track_buffer
        args.cmc_method = cmc_method

        self.tracker = McByteTracker(args, save_folder=save_folder, frame_rate=frame_rate)
        self.frame_id = 0
        self._prev_track_ids: Set[int] = set()

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
        test_size: tuple = (640, 640),
        prediction_mask: Optional[np.ndarray] = None,
        tracklet_mask_dict: Optional[Dict] = None,
        mask_avg_prob_dict: Optional[Dict] = None,
        vis_type: str = "basic",
    ) -> TrackerOutput:
        """
        Update McByteTracker with detections.

        This adapter wraps the original McByteTracker.update() method.
        """
        self.frame_id += 1

        # Convert detections to expected format if needed
        if detections is None or len(detections) == 0:
            det_tensor = np.zeros((0, 5))
        else:
            det_tensor = detections

        # Call original tracker
        (
            online_targets_raw,
            removed_tracks_ids,
            new_confirmed_tracks_raw,
            _,
            _
        ) = self.tracker.update(
            det_tensor,
            [img_info['height'], img_info['width']],
            test_size,
            prediction_mask=prediction_mask,
            tracklet_mask_dict=tracklet_mask_dict if tracklet_mask_dict else {},
            mask_avg_prob_dict=mask_avg_prob_dict if mask_avg_prob_dict else {},
            frame_img=img_info['raw_img'],
            vis_type=vis_type,
            dets_from_file=True,  # Detections already scaled
        )

        # Convert to Track objects
        online_targets = []
        for t in online_targets_raw:
            track = Track(
                track_id=t.track_id,
                tlwh=np.array(t.tlwh),
                last_det_tlwh=np.array(t.last_det_tlwh),
                score=t.score,
            )
            online_targets.append(track)

        new_tracks = []
        for t in new_confirmed_tracks_raw:
            track = Track(
                track_id=t.track_id,
                tlwh=np.array(t.tlwh),
                last_det_tlwh=np.array(t.last_det_tlwh),
                score=t.score,
            )
            new_tracks.append(track)

        return TrackerOutput(online_targets, removed_tracks_ids, new_tracks)

    def reset(self):
        """Reset tracker for new video."""
        self.tracker.tracked_stracks = []
        self.tracker.lost_stracks = []
        self.tracker.removed_stracks = []
        self.tracker.frame_id = 0
        self.frame_id = 0
        self._prev_track_ids = set()


class ByteTrackAdapter(BaseTrackerInterface):
    """
    Adapter for standard ByteTrack (without mask enhancement).

    Uses the same underlying tracker but without mask-based cost adjustment.
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        track_buffer: int = 30,
        frame_rate: int = 30,
        save_folder: str = ".",
        cmc_method: str = "orb",
    ):
        # Reuse McByteTracker but we'll call it without mask info
        self.adapter = McByteTrackerAdapter(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            save_folder=save_folder,
            cmc_method=cmc_method,
        )

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
        test_size: tuple = (640, 640),
    ) -> TrackerOutput:
        """Update ByteTrack without mask information."""
        # Call McByte adapter but without mask info (standard ByteTrack behavior)
        return self.adapter.update(
            detections,
            img_info,
            test_size=test_size,
            prediction_mask=None,
            tracklet_mask_dict={},
            mask_avg_prob_dict={},
            vis_type="basic",
        )

    def reset(self):
        self.adapter.reset()


class SORTAdapter(BaseTrackerInterface):
    """
    Adapter for SORT tracker.

    Simple Online and Realtime Tracking using Kalman filter + Hungarian matching.
    Requires: pip install filterpy
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, Track] = {}
        self._sort = None
        self._init_sort()

    def _init_sort(self):
        """Initialize SORT tracker."""
        try:
            from sort import Sort
            self._sort = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
        except ImportError:
            # Fallback to simple implementation if sort package not available
            self._sort = None

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
    ) -> TrackerOutput:
        """Update SORT tracker."""
        if self._sort is None:
            raise ImportError(
                "SORT package not found. Install with: pip install sort-tracker"
            )

        # SORT expects [x1, y1, x2, y2, score]
        if detections is None or len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = detections if isinstance(detections, np.ndarray) else detections.cpu().numpy()

        # Run SORT - returns [x1, y1, x2, y2, track_id]
        tracks = self._sort.update(dets)

        online_targets = []
        current_ids = set()

        for t in tracks:
            x1, y1, x2, y2, track_id = t[:5]
            track_id = int(track_id)
            current_ids.add(track_id)

            # Convert to tlwh
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1])

            track = Track(
                track_id=track_id,
                tlwh=tlwh,
                last_det_tlwh=tlwh.copy(),
                score=1.0  # SORT doesn't preserve detection scores
            )
            online_targets.append(track)

        # Determine new and removed tracks
        prev_ids = set(self._tracks.keys())
        new_track_ids = current_ids - prev_ids
        removed_track_ids = list(prev_ids - current_ids)
        new_tracks = [t for t in online_targets if t.track_id in new_track_ids]

        self._tracks = {t.track_id: t for t in online_targets}

        return TrackerOutput(online_targets, removed_track_ids, new_tracks)

    def reset(self):
        """Reset SORT for new video."""
        self._tracks = {}
        self._init_sort()


class DeepSORTAdapter(BaseTrackerInterface):
    """
    Adapter for DeepSORT tracker.

    SORT with deep appearance features (ReID).
    Requires: pip install deep-sort-realtime
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder: str = "mobilenet",
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.embedder = embedder
        self._tracker = None
        self._tracks: Dict[int, Track] = {}
        self._init_tracker()

    def _init_tracker(self):
        """Initialize DeepSORT tracker."""
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                embedder=self.embedder,
            )
        except ImportError:
            self._tracker = None

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
    ) -> TrackerOutput:
        """Update DeepSORT tracker."""
        if self._tracker is None:
            raise ImportError(
                "deep-sort-realtime package not found. "
                "Install with: pip install deep-sort-realtime"
            )

        frame = img_info['raw_img']

        # Format detections for DeepSORT: [[x1, y1, w, h, confidence], ...]
        if detections is None or len(detections) == 0:
            bbs = []
        else:
            dets = detections if isinstance(detections, np.ndarray) else detections.cpu().numpy()
            bbs = []
            for d in dets:
                x1, y1, x2, y2, conf = d
                w = x2 - x1
                h = y2 - y1
                bbs.append(([x1, y1, w, h], conf, 'person'))  # DeepSORT format

        # Update tracker
        tracks = self._tracker.update_tracks(bbs, frame=frame)

        online_targets = []
        current_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            current_ids.add(track_id)

            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            tlwh = np.array([
                ltrb[0],
                ltrb[1],
                ltrb[2] - ltrb[0],
                ltrb[3] - ltrb[1]
            ])

            t = Track(
                track_id=track_id,
                tlwh=tlwh,
                last_det_tlwh=tlwh.copy(),
                score=track.det_conf if track.det_conf else 1.0,
            )
            online_targets.append(t)

        prev_ids = set(self._tracks.keys())
        new_track_ids = current_ids - prev_ids
        removed_track_ids = list(prev_ids - current_ids)
        new_tracks = [t for t in online_targets if t.track_id in new_track_ids]

        self._tracks = {t.track_id: t for t in online_targets}

        return TrackerOutput(online_targets, removed_track_ids, new_tracks)

    def reset(self):
        """Reset DeepSORT for new video."""
        self._tracks = {}
        self._init_tracker()


class BoTSORTAdapter(BaseTrackerInterface):
    """
    Adapter for BoT-SORT tracker.

    ByteTrack with camera motion compensation and ReID features.
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        track_buffer: int = 30,
        frame_rate: int = 30,
        cmc_method: str = "orb",
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.cmc_method = cmc_method
        self._tracker = None
        self._tracks: Dict[int, Track] = {}

        # BoT-SORT is similar to ByteTrack with GMC, reuse ByteTrack adapter
        # which already has GMC support
        self._bytetrack = ByteTrackAdapter(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            cmc_method=cmc_method,
        )

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
        test_size: tuple = (640, 640),
    ) -> TrackerOutput:
        """Update BoT-SORT tracker."""
        # BoT-SORT is essentially ByteTrack + CMC + optional ReID
        # The McByteTracker already includes GMC, so we use that
        return self._bytetrack.update(detections, img_info, test_size)

    def reset(self):
        """Reset BoT-SORT for new video."""
        self._bytetrack.reset()
        self._tracks = {}


class OCSORTAdapter(BaseTrackerInterface):
    """
    Adapter for OC-SORT tracker.

    Observation-Centric SORT - handles occlusions better.
    Requires: pip install ocsort
    """

    def __init__(
        self,
        det_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        inertia: float = 0.2,
    ):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self._tracker = None
        self._tracks: Dict[int, Track] = {}
        self._init_tracker()

    def _init_tracker(self):
        """Initialize OC-SORT tracker."""
        try:
            from ocsort import OCSort
            self._tracker = OCSort(
                det_thresh=self.det_thresh,
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
                delta_t=self.delta_t,
                inertia=self.inertia,
            )
        except ImportError:
            self._tracker = None

    def update(
        self,
        detections: np.ndarray,
        img_info: Dict[str, Any],
    ) -> TrackerOutput:
        """Update OC-SORT tracker."""
        if self._tracker is None:
            raise ImportError(
                "ocsort package not found. Install with: pip install ocsort"
            )

        # OC-SORT expects [x1, y1, x2, y2, score]
        if detections is None or len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = detections if isinstance(detections, np.ndarray) else detections.cpu().numpy()

        # Run OC-SORT
        tracks = self._tracker.update(dets, [img_info['height'], img_info['width']])

        online_targets = []
        current_ids = set()

        for t in tracks:
            x1, y1, x2, y2, track_id = t[:5]
            track_id = int(track_id)
            current_ids.add(track_id)

            tlwh = np.array([x1, y1, x2 - x1, y2 - y1])

            track = Track(
                track_id=track_id,
                tlwh=tlwh,
                last_det_tlwh=tlwh.copy(),
                score=1.0
            )
            online_targets.append(track)

        prev_ids = set(self._tracks.keys())
        new_track_ids = current_ids - prev_ids
        removed_track_ids = list(prev_ids - current_ids)
        new_tracks = [t for t in online_targets if t.track_id in new_track_ids]

        self._tracks = {t.track_id: t for t in online_targets}

        return TrackerOutput(online_targets, removed_track_ids, new_tracks)

    def reset(self):
        """Reset OC-SORT for new video."""
        self._tracks = {}
        self._init_tracker()


# Registry of available trackers
TRACKER_REGISTRY = {
    "mcbyte": McByteTrackerAdapter,
    "bytetrack": ByteTrackAdapter,
    "sort": SORTAdapter,
    "deepsort": DeepSORTAdapter,
    "botsort": BoTSORTAdapter,
    "ocsort": OCSORTAdapter,
}


def create_tracker(
    tracker_type: str = "mcbyte",
    track_thresh: float = 0.6,
    track_buffer: int = 30,
    frame_rate: int = 30,
    save_folder: str = ".",
    **kwargs
) -> BaseTrackerInterface:
    """
    Factory function to create a tracker by name.

    Args:
        tracker_type: One of "mcbyte", "bytetrack", "sort", "deepsort", "botsort", "ocsort"
        track_thresh: Detection confidence threshold
        track_buffer: Frames to keep lost tracks
        frame_rate: Video frame rate
        save_folder: Folder for logging (used by McByte)
        **kwargs: Additional tracker-specific arguments

    Returns:
        Configured tracker instance

    Raises:
        ValueError: If tracker_type is not recognized
    """
    tracker_type = tracker_type.lower()

    if tracker_type not in TRACKER_REGISTRY:
        available = ", ".join(TRACKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown tracker type: {tracker_type}. Available: {available}"
        )

    tracker_cls = TRACKER_REGISTRY[tracker_type]

    # Build arguments based on tracker type
    if tracker_type in ["mcbyte", "bytetrack", "botsort"]:
        return tracker_cls(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            save_folder=save_folder,
            **kwargs
        )
    elif tracker_type == "sort":
        return tracker_cls(
            max_age=track_buffer,
            min_hits=kwargs.get("min_hits", 3),
            iou_threshold=kwargs.get("iou_threshold", 0.3),
        )
    elif tracker_type == "deepsort":
        return tracker_cls(
            max_age=track_buffer,
            n_init=kwargs.get("n_init", 3),
            max_iou_distance=kwargs.get("max_iou_distance", 0.7),
            embedder=kwargs.get("embedder", "mobilenet"),
        )
    elif tracker_type == "ocsort":
        return tracker_cls(
            det_thresh=track_thresh,
            max_age=track_buffer,
            min_hits=kwargs.get("min_hits", 3),
            iou_threshold=kwargs.get("iou_threshold", 0.3),
            delta_t=kwargs.get("delta_t", 3),
            inertia=kwargs.get("inertia", 0.2),
        )

    return tracker_cls(**kwargs)


def list_available_trackers() -> List[str]:
    """Return list of available tracker types."""
    return list(TRACKER_REGISTRY.keys())
