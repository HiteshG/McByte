"""
Class configuration for multi-class tracking.

This module defines which classes are detected, which to track,
colors for visualization, and special classes (e.g., small objects
where only max-confidence detection is kept).
"""
from enum import IntEnum


class TrackableClass(IntEnum):
    """Class IDs as detected by YOLOX model."""
    CENTER_ICE = 0
    FACEOFF = 1
    GOALPOST = 2
    GOALTENDER = 3
    PLAYER = 4
    PUCK = 5
    REFEREE = 6


# Human-readable class names
CLASS_NAMES = {
    TrackableClass.CENTER_ICE: "Center Ice",
    TrackableClass.FACEOFF: "Faceoff",
    TrackableClass.GOALPOST: "Goalpost",
    TrackableClass.GOALTENDER: "Goaltender",
    TrackableClass.PLAYER: "Player",
    TrackableClass.PUCK: "Puck",
    TrackableClass.REFEREE: "Referee",
}

# Classes to track (filter out others like Center Ice, Faceoff, Goalpost)
# Using int() to ensure compatibility with numpy operations
TRACK_CLASSES = [
    int(TrackableClass.PLAYER),       # 4
    int(TrackableClass.PUCK),         # 5
    int(TrackableClass.REFEREE),      # 6
    int(TrackableClass.GOALTENDER),   # 3
]

# Class colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    TrackableClass.GOALTENDER: (255, 0, 0),    # Blue
    TrackableClass.PLAYER: (0, 255, 0),        # Green
    TrackableClass.PUCK: (0, 255, 255),        # Yellow
    TrackableClass.REFEREE: (0, 0, 255),       # Red
}

# Special classes: keep only max-confidence detection per frame
# Useful for small objects like puck where multiple false positives are common
SPECIAL_CLASSES = [
    int(TrackableClass.PUCK),  # 5
]

# Number of classes the model was trained on
NUM_CLASSES = 7
