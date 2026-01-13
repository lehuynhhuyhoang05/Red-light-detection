"""
Package initializer for src module
"""

__version__ = "1.0.0"
__author__ = "ITS Research Team"

from .detector import Detection, BaseDetector, create_detector
from .tracker import TrackedObject, ObjectTracker
from .violation_logic import Violation, ViolationDetector
from .utils import load_config, setup_logging

__all__ = [
    'Detection',
    'BaseDetector',
    'create_detector',
    'TrackedObject',
    'ObjectTracker',
    'Violation',
    'ViolationDetector',
    'load_config',
    'setup_logging',
]
