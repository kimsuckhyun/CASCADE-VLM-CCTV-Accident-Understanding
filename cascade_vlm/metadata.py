#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata loading and scene-conditioned prompt block construction.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import re
import pandas as pd
import numpy as np


def load_metadata(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        logger.warning(f"메타데이터 CSV 없음: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"메타데이터 로드 완료: {csv_path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"메타데이터 로드 실패: {e}")
        return None


def get_video_key(video_path):
    return Path(video_path).name


def get_metadata_for_video(meta_df, video_path):
    if meta_df is None or "path" not in meta_df.columns:
        return {}
    key = get_video_key(video_path)
    rows = meta_df[meta_df["path"].astype(str) == key]
    if len(rows) == 0:
        rows = meta_df[meta_df["path"].astype(str).apply(lambda x: Path(x).name == key)]
    if len(rows) == 0:
        logger.warning(f"메타데이터 매칭 실패: {key}")
        return {}
    return rows.iloc[0].to_dict()


def clean_meta_value(v):
    try:
        if pd.isna(v):
            return "unknown"
    except Exception:
        pass
    return str(v)


def meta_number(meta, keys, default=None, cast=float):
    if not meta:
        return default
    for k in keys:
        if k in meta:
            try:
                v = meta[k]
                if pd.isna(v):
                    continue
                return cast(v)
            except Exception:
                continue
    return default


def build_metadata_prompt_block(meta):
    if not meta:
        return """
Metadata context:
- scene_layout: unknown
- weather: unknown
- day_time: unknown
- quality: unknown
- region: unknown
- duration: unknown
- height: unknown
- width: unknown
- no_frames: unknown

Use metadata only as contextual prior.
Do NOT decide accident type, time, or point from metadata alone.
Final decisions must be based on visible evidence in frames.
"""

    scene_layout = clean_meta_value(meta.get("scene_layout", "unknown")).lower()
    weather = clean_meta_value(meta.get("weather", "unknown")).lower()
    day_time = clean_meta_value(meta.get("day_time", "unknown")).lower()
    quality = clean_meta_value(meta.get("quality", "unknown")).lower()
    region = clean_meta_value(meta.get("region", "unknown"))
    meta_dur = clean_meta_value(meta.get("duration", meta.get("video_duration", "unknown")))
    meta_h = clean_meta_value(meta.get("height", meta.get("h", "unknown")))
    meta_w = clean_meta_value(meta.get("width", meta.get("w", "unknown")))
    meta_nf = clean_meta_value(meta.get("no_frames", meta.get("num_frames", "unknown")))

    extra_rules = []

    if day_time in ["night", "dark", "evening"]:
        extra_rules.append("- Night scene: rely more on motion continuity, trajectory change, tiny motion discontinuity, sudden stop, spin, and consistent pre/post-impact behavior.")
        extra_rules.append("- Do NOT treat headlight bloom, flare, or a bright patch alone as accident evidence.")

    if weather in ["rain", "rainy", "snow", "snowy", "fog", "foggy", "wet"]:
        extra_rules.append("- Adverse weather: ignore reflections, wet-road shine, spray, rain streaks, snow streaks, and glare unless real crash behavior is visible across consecutive frames.")
        extra_rules.append("- Do NOT mistake splash or spray for first contact.")

    if quality in ["low", "poor", "very_low", "bad"]:
        extra_rules.append("- Low-quality CCTV: do not rely on one blurry frame; use multiple consecutive frames and before/after motion consistency.")
        extra_rules.append("- Compression blocks, smearing, and motion blur are not collision evidence by themselves.")

    if "ramp" in scene_layout or "intersection" in scene_layout or "roundabout" in scene_layout:
        extra_rules.append("- Complex road layout: do NOT confuse normal turning, merge, or lane weaving with collision.")
        extra_rules.append("- Prefer the true small first-contact point over a large unrelated central object.")

    if not extra_rules:
        extra_rules.append("- No extra visibility rule from metadata.")

    return f"""
Metadata context:
- scene_layout: {scene_layout}
- weather: {weather}
- day_time: {day_time}
- quality: {quality}
- region: {region}
- duration: {meta_dur}
- height: {meta_h}
- width: {meta_w}
- no_frames: {meta_nf}

Environment-specific rules:
{chr(10).join(extra_rules)}

Use metadata only as contextual prior.
Do NOT decide accident type, time, or point from metadata alone.
Final decisions must be based on visible evidence in frames.
"""


def metadata_type_prior(meta):
    if not meta:
        return None
    scene = clean_meta_value(meta.get("scene_layout", "")).lower()
    if any(k in scene for k in ["intersection", "crossroad", "junction", "signal"]):
        return {"t-bone": 0.45, "rear-end": 0.25, "single": 0.15, "sideswipe": 0.10, "head-on": 0.05}
    elif any(k in scene for k in ["highway", "expressway", "freeway", "straight"]):
        return {"rear-end": 0.35, "single": 0.35, "sideswipe": 0.15, "head-on": 0.10, "t-bone": 0.05}
    elif any(k in scene for k in ["parking", "lot", "garage"]):
        return {"single": 0.30, "sideswipe": 0.25, "rear-end": 0.20, "t-bone": 0.15, "head-on": 0.10}
    elif any(k in scene for k in ["ramp", "curve", "bend"]):
        return {"single": 0.45, "sideswipe": 0.25, "rear-end": 0.20, "t-bone": 0.05, "head-on": 0.05}
    return None


# ─────────────────────────────────────────────
