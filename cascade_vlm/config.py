#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for CASCADE-VLM.

All hyperparameters, thresholds, and default paths are defined here.
"""

# ─────────────────────────────────────────────
# VLM server
# ─────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-27B"
MAX_IMAGES_PER_PROMPT = 128


# ─────────────────────────────────────────────
# Default paths (override via CLI arguments)
# ─────────────────────────────────────────────
DEFAULT_VIDEO_DIR = "/path/to/cctv/videos"
DEFAULT_METADATA_CSV = "/path/to/test_metadata.csv"


# ─────────────────────────────────────────────
# Accident type labels (5 ACCIDENT classes)
# ─────────────────────────────────────────────
TYPE_LIST = ["single", "rear-end", "t-bone", "sideswipe", "head-on"]
TYPE_SET = set(TYPE_LIST)


# ─────────────────────────────────────────────
# Top-K candidates after coarse stage
# ─────────────────────────────────────────────
COARSE_TOP_K = 3


# ─────────────────────────────────────────────
# Grid localization (Stage 3) settings
# ─────────────────────────────────────────────
GRID_ROWS = 3
GRID_COLS = 3
GRID_ROUNDS = 3


# ─────────────────────────────────────────────
# Frame resize (min side / max side caps)
# ─────────────────────────────────────────────
COARSE_MIN_SIDE = 520
COARSE_MAX_SIDE = 960
REFINE_MIN_SIDE = 560
REFINE_MAX_SIDE = 1024


# ─────────────────────────────────────────────
# Spatial region tiling (Stage 1)
# ─────────────────────────────────────────────
SPATIAL_BASE_DIV = 2          # 2x2 base grid
SPATIAL_STRIDE_RATIO = 0.5    # half-region stride → 3x3 = 9 regions


# ─────────────────────────────────────────────
# Candidate clustering thresholds
# ─────────────────────────────────────────────
CLUSTER_SCORE_THRESHOLD = 70.0   # merge regions with score >= 70
CLUSTER_IOU_THRESHOLD = 0.15     # merge if spatial IoU >= 0.15
