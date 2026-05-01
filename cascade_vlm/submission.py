#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submission CSV update utilities.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
from .parsing import force_valid_type


def load_submission_template(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"submission template not found: {path}")
    df = pd.read_csv(path)
    need_cols = ["path", "accident_time", "center_x", "center_y", "type"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"submission template missing column: {c}")
    return df


def find_submission_row_index(df, video_path_value):
    key = Path(video_path_value).name
    exact = df.index[df["path"].astype(str) == video_path_value].tolist()
    if exact:
        return exact[0]
    by_name = df.index[df["path"].astype(str).apply(lambda x: Path(x).name == key)].tolist()
    if by_name:
        return by_name[0]
    return None


def result_to_submission_fields(video_path_value, result):
    vi = result["video_info"]
    w, h = max(1, int(vi["w"])), max(1, int(vi["h"]))
    pt = result.get("accident_point") or {"x": w // 2, "y": h // 2}
    center_x = float(pt["x"]) / float(w)
    center_y = float(pt["y"]) / float(h)
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    return {
        "path": video_path_value,
        "accident_time": float(result["peak"]),
        "center_x": center_x,
        "center_y": center_y,
        "type": force_valid_type(result.get("accident_type"), fallback="single"),
    }


def update_submission_row(df, video_path_value, result):
    row_idx = find_submission_row_index(df, video_path_value)
    pred = result_to_submission_fields(video_path_value, result)
    if row_idx is None:
        df.loc[len(df)] = pred
    else:
        df.at[row_idx, "accident_time"] = pred["accident_time"]
        df.at[row_idx, "center_x"] = pred["center_x"]
        df.at[row_idx, "center_y"] = pred["center_y"]
        df.at[row_idx, "type"] = pred["type"]
    return df


def save_submission_atomic(df, out_path):
    tmp = str(Path(out_path).with_suffix(".tmp.csv"))
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


# ─────────────────────────────────────────────
