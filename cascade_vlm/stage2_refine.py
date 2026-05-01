#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Cascaded Temporal Refinement and Scene-conditioned Type Verification.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import json
import logging
import cv2
import numpy as np

from .config import (
    TYPE_LIST, REFINE_MIN_SIDE, REFINE_MAX_SIDE, MAX_IMAGES_PER_PROMPT,
)

logger = logging.getLogger(__name__)
from .metadata import clean_meta_value, metadata_type_prior
from .parsing import (
    _extract_response_text, ask_frames, clamp_time, force_valid_type,
    nearest_timestamp, normalize_refine_result, safe_json_loads,
    sanitize_type, timestamp_mapping_text,
)
from .video_utils import region_to_text, resize_keep_ratio_min_side, sec_to_frame_idx, stamp_frame


def expand_region(region, video_w, video_h, expand_ratio=0.20):
    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    rw, rh = x2 - x1, y2 - y1
    pad_w = int(round(rw * expand_ratio))
    pad_h = int(round(rh * expand_ratio))
    return {
        "x1": max(0, x1 - pad_w),
        "y1": max(0, y1 - pad_h),
        "x2": min(video_w, x2 + pad_w),
        "y2": min(video_h, y2 + pad_h),
        "w": min(video_w, x2 + pad_w) - max(0, x1 - pad_w),
        "h": min(video_h, y2 + pad_h) - max(0, y1 - pad_h),
        "region_id": region.get("region_id", -1),
        "grid_pos": region.get("grid_pos", "expanded"),
    }


def extract_region_frames_uniform(path, region, fps, start, end, max_n=32, min_side=560, max_side=1024, do_stamp=True):
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / vfps if vfps > 0 else 0.0
    end = min(end, dur)
    gap = 1.0 / fps
    sec = start
    raw = []

    while sec <= end + 1e-9:
        if len(raw) >= max_n:
            break
        idx = sec_to_frame_idx(sec, vfps, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok or fr is None:
            sec += gap
            continue
        x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        crop = fr[y1:y2, x1:x2]
        raw.append({"frame": crop, "sec": round(float(sec), 3)})
        sec += gap
    cap.release()

    out = []
    total_frames = len(raw)
    for i, item in enumerate(raw):
        fr_small = resize_keep_ratio_min_side(item["frame"], min_side, max_side)
        if do_stamp:
            fr_small = stamp_frame(fr_small, i, total_frames, item["sec"])
        ok, buf = cv2.imencode(".jpg", fr_small, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            continue
        out.append({"frame": item["frame"], "sec": item["sec"], "b64": base64.b64encode(buf).decode()})
    return out


def refine_candidate(client, vpath, dur, rough, coarse_window, meta_block, video_w, video_h):
    coarse_type = force_valid_type(coarse_window.get("accident_type"), fallback="single")
    coarse_score = float(coarse_window.get("score", 0))
    best_desc = coarse_window.get("description", "")
    base_region = coarse_window.get("region_union", coarse_window["region"])
    region = expand_region(base_region, video_w, video_h, expand_ratio=0.20)
    region_text = region_to_text(region, video_w, video_h)

    mid_fps = 8.0
    mid_half = 1.9
    s2 = max(0.0, rough - mid_half)
    e2 = min(dur, rough + mid_half)
    f2 = extract_region_frames_uniform(vpath, region, fps=mid_fps, start=s2, end=e2, max_n=32,
                                       min_side=REFINE_MIN_SIDE, max_side=REFINE_MAX_SIDE, do_stamp=True)

    if f2:
        actual_s2 = f2[0]["sec"]
        actual_e2 = f2[-1]["sec"]
        prompt = f"""
{meta_block}

You are given sequential LOCAL CROP CCTV frames near the estimated accident time.
These frames come from:
- {region_text}

Frames are in CHRONOLOGICAL ORDER with "[N/Total] T.TTs" labels.

{VEHICLE_ACCIDENT_DEFINITION}
{ACCIDENT_TYPE_5_DEFINITION}
{LOW_VISIBILITY_RULES}
{LENS_OBSTRUCTION_RULES}
{DISTANT_SMALL_ACCIDENT_RULES}
{FALSE_POSITIVE_RULES}

Refinement stage:
- Actual provided frame span: {actual_s2:.2f}s to {actual_e2:.2f}s
- Sample rate: {mid_fps:.2f} fps
- Number of frames: {len(f2)}

Task:
- start_sec: when dangerous crash event clearly begins
- peak_sec: timestamp closest to the FIRST physical contact moment
- end_sec: when the immediate accident phase settles
- accident_type: MUST choose exactly one of {TYPE_LIST}
- confidence: 0~100

Important:
- peak_sec MUST be one of the provided timestamps
- Track the FIRST contact in this local crop
- If there is visible vehicle-to-vehicle contact, single is forbidden
- Do NOT output unknown

Respond ONLY in JSON:
{{
  "start_sec": <float>,
  "peak_sec": <float>,
  "end_sec": <float>,
  "accident_type": "single" or "rear-end" or "t-bone" or "sideswipe" or "head-on",
  "confidence": <0-100>,
  "description": "<detailed local-region accident description>"
}}
"""
        raw = ask_frames(client, f2, prompt, max_tok=1700)
        r2 = normalize_refine_result(safe_json_loads(raw), f2, rough, coarse_type)
    else:
        r2 = {
            "start_sec": max(0.0, rough - 0.5),
            "peak_sec": rough,
            "end_sec": min(dur, rough + 1.0),
            "description": best_desc,
            "accident_type": coarse_type,
            "confidence": 0.0,
        }

    mid_peak = clamp_time(r2["peak_sec"], dur)
    if mid_peak is None:
        mid_peak = rough

    fine_fps = 12.0
    fine_half = 1.25
    s3 = max(0.0, mid_peak - fine_half)
    e3 = min(dur, mid_peak + fine_half)
    f3 = extract_region_frames_uniform(vpath, region, fps=fine_fps, start=s3, end=e3, max_n=32,
                                       min_side=700, max_side=1200, do_stamp=True)

    if f3:
        actual_s3 = f3[0]["sec"]
        actual_e3 = f3[-1]["sec"]
        prompt = f"""
{meta_block}

You are in the FINAL local-region time localization stage for a CCTV vehicle accident.
These frames come from:
- {region_text}

Frames are in CHRONOLOGICAL ORDER with "[N/Total] T.TTs" labels.

{VEHICLE_ACCIDENT_DEFINITION}
{ACCIDENT_TYPE_5_DEFINITION}
{LOW_VISIBILITY_RULES}
{LENS_OBSTRUCTION_RULES}
{DISTANT_SMALL_ACCIDENT_RULES}
{FALSE_POSITIVE_RULES}

Final refinement info:
- Actual provided frame span: {actual_s3:.2f}s to {actual_e3:.2f}s
- Sample rate: {fine_fps:.2f} fps
- Number of frames: {len(f3)}

Goal:
Select the single provided timestamp closest to the FIRST physical contact moment within this local region.

Important:
- peak_sec MUST be exactly one of the provided timestamps
- accident_type MUST be exactly one of {TYPE_LIST}
- Compare each frame to the NEXT frame
- If there is visible vehicle-to-vehicle contact, single is forbidden
- Do NOT output unknown

Respond ONLY in JSON:
{{
  "start_sec": <float>,
  "peak_sec": <float>,
  "end_sec": <float>,
  "accident_type": "single" or "rear-end" or "t-bone" or "sideswipe" or "head-on",
  "confidence": <0-100>,
  "description": "<final local-region detailed description>"
}}
"""
        raw = ask_frames(client, f3, prompt, max_tok=1800)
        r3 = normalize_refine_result(safe_json_loads(raw), f3, mid_peak, r2["accident_type"])
    else:
        r3 = {
            "start_sec": max(0.0, mid_peak - 0.4),
            "peak_sec": mid_peak,
            "end_sec": min(dur, mid_peak + 0.8),
            "description": r2.get("description", best_desc),
            "accident_type": r2.get("accident_type", coarse_type),
            "confidence": 0.0,
        }

    peak = clamp_time(r3.get("peak_sec", mid_peak), dur)
    if peak is None:
        peak = mid_peak
    peak = nearest_timestamp(peak, f3 if f3 else f2 if f2 else [])

    start = clamp_time(r3.get("start_sec", peak - 0.5), dur)
    if start is None:
        start = max(0.0, peak - 0.5)

    end = clamp_time(r3.get("end_sec", peak + 1.0), dur)
    if end is None:
        end = min(dur, peak + 1.0)

    if start > peak:
        start = max(0.0, peak - 0.3)
    if end < peak:
        end = min(dur, peak + 0.6)

    final_type = force_valid_type(r3.get("accident_type"), r2.get("accident_type"), coarse_type, fallback="single")
    desc = r3.get("description", "") or r2.get("description", "") or best_desc

    return {
        "coarse_score": coarse_score,
        "mid_score": float(r2.get("confidence", 0)),
        "final_score": float(r3.get("confidence", 0)),
        "accident_type": final_type,
        "start": round(start, 3),
        "peak": round(peak, 3),
        "end": round(end, 3),
        "description": desc,
        "coarse_window": coarse_window,
        "refine_region": region,
        "_type_chain": [coarse_type, r2.get("accident_type", coarse_type), r3.get("accident_type", coarse_type)],
    }


TYPE_VERIFY_SYSTEM = f"""You are an expert CCTV vehicle accident TYPE classifier.

{ACCIDENT_TYPE_5_DEFINITION}

{LOW_VISIBILITY_RULES}

{LENS_OBSTRUCTION_RULES}

Rules:
- Output ONLY one valid JSON object.
- Do NOT output reasoning, markdown, or any text before/after JSON.
"""


def verify_accident_type(client, vpath, peak, dur, current_type, meta_block, meta, region=None):
    offsets = [-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6]
    frames = []
    cap = cv2.VideoCapture(vpath)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_list = []
    for dt in offsets:
        sec = max(0.0, min(dur, peak + dt))
        idx = sec_to_frame_idx(sec, vfps, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok and fr is not None:
            if region is not None:
                x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
                fr = fr[y1:y2, x1:x2]
            raw_list.append({"frame": fr, "sec": round(sec, 3)})
    cap.release()

    total_f = len(raw_list)
    for i, item in enumerate(raw_list):
        fr = resize_keep_ratio_min_side(item["frame"], 600, 1000)
        fr = stamp_frame(fr, i, total_f, item["sec"])
        ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok2:
            frames.append({"sec": item["sec"], "b64": base64.b64encode(buf).decode()})

    if len(frames) < 3:
        return current_type

    scene = clean_meta_value(meta.get("scene_layout", "")).lower() if meta else ""
    scene_hint = ""
    if any(k in scene for k in ["intersection", "crossroad", "junction", "signal"]):
        scene_hint = "NOTE: This is an intersection scene. Vehicles from cross-streets hitting the side of another vehicle = t-bone (NOT rear-end)."
    elif any(k in scene for k in ["highway", "expressway", "freeway"]):
        scene_hint = "NOTE: This is a highway/expressway scene. Same-direction rear collisions are common."

    ts = timestamp_mapping_text(frames)

    prompt = f"""
{meta_block}

You are verifying the accident TYPE classification only.
The accident time is already determined at {peak:.2f}s.
Current type estimate: {current_type}
{scene_hint}

Frames are in CHRONOLOGICAL ORDER with "[N/Total] T.TTs" labels.
Frame timestamps: [{ts}]

These frames show the moment just before, during, and after the first physical contact.

CRITICAL classification rules:
- Look at the APPROACH DIRECTIONS of the vehicles before impact:
  * Same direction → rear-end
  * Perpendicular → t-bone
  * Opposite direction → head-on
  * Same direction, side-to-side contact → sideswipe
  * No other vehicle involved → single

Respond ONLY in JSON:
{{
  "accident_type": "single" or "rear-end" or "t-bone" or "sideswipe" or "head-on",
  "approach_directions": "<describe how each vehicle approaches>",
  "contact_parts": "<which part of each vehicle makes first contact>",
  "angle": "<estimated angle between vehicles at impact>"
}}
"""

    content = []
    for fd in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fd['b64']}"}})
    content.append({"type": "text", "text": prompt + "\nReturn ONLY one raw JSON object. Your first character must be '{' and your last character must be '}'."})

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": TYPE_VERIFY_SYSTEM}, {"role": "user", "content": content}],
        max_tokens=1200,
        temperature=0.0,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = _extract_response_text(r)
    result = safe_json_loads(raw)
    new_type = sanitize_type(result.get("accident_type"))
    if new_type and new_type in TYPE_SET:
        logger.info(f"  [TypeVerify] {current_type} → {new_type}")
        return new_type
    return current_type


def resolve_type_with_prior(type_chain, verified_type, meta):
    if verified_type in type_chain:
        return verified_type
    chain_set = set(type_chain)
    if len(chain_set) == 1:
        chain_unanimous = list(chain_set)[0]
        prior = metadata_type_prior(meta)
        if prior:
            score_chain = prior.get(chain_unanimous, 0)
            score_verified = prior.get(verified_type, 0)
            if score_verified > score_chain * 1.5:
                return verified_type
        return chain_unanimous
    return verified_type


# ─────────────────────────────────────────────
