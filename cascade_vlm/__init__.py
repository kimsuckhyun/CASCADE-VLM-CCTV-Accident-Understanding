"""
CASCADE-VLM: A Zero-Shot Cascade Pipeline for CCTV Accident Understanding.

CVPR 2026 ACCIDENT Challenge submission.
"""

__version__ = "1.0.0"

from .client import get_client, check_server
from .pipeline import detect
from .metadata import load_metadata, get_metadata_for_video

__all__ = [
    "get_client",
    "check_server",
    "detect",
    "load_metadata",
    "get_metadata_for_video",
]
