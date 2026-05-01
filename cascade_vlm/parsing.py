#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Response parsing and result normalization.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import json
import re
import logging

from .config import TYPE_LIST, TYPE_SET

logger = logging.getLogger(__name__)


def _extract_response_text(r):
    msg = r.choices[0].message
    txt = ""
    if getattr(msg, "content", None):
        txt = msg.content
    elif hasattr(msg, "reasoning_content") and msg.reasoning_content:
        txt = msg.reasoning_content
    else:
        txt = str(msg)
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        return m.group(0)
    return txt


def safe_json_loads(text):
    if not text:
        return {}
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"JSON 파싱 실패: {e}\n{text[:800]}")
        return {}


def timestamp_mapping_text(frames):
    return ", ".join(f"Frame{i+1}={fd['sec']}s" for i, fd in enumerate(frames))


def ask_frames(client, frames, prompt, max_tok=2048, max_images=MAX_IMAGES_PER_PROMPT):
    if len(frames) > max_images:
        frames = frames[:max_images]

    content = []
    for fd in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fd['b64']}"}
        })

    ts = timestamp_mapping_text(frames)
    full_prompt = (
        prompt
        + "\n\n"
        + f"Frame timestamp mapping (chronological order): [{ts}]\n"
        + f"Total frames: {len(frames)}, provided in strict time order (first image = earliest, last image = latest).\n"
        + "Return ONLY one raw JSON object. Your first character must be '{' and your last character must be '}'."
    )
    content.append({"type": "text", "text": full_prompt})

    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tok,
        temperature=0.0,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return _extract_response_text(r)


def clamp_time(t, duration):
    try:
        t = float(t)
    except Exception:
        return None
    return max(0.0, min(duration, t))


def nearest_timestamp(target_t, frames):
    if not frames:
        return target_t
    return min((fd["sec"] for fd in frames), key=lambda x: abs(x - target_t))


def sanitize_type(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    alias = {
        "rear_end": "rear-end",
        "rear end": "rear-end",
        "head_on": "head-on",
        "head on": "head-on",
        "side impact": "t-bone",
        "side-impact": "t-bone",
        "broadside": "t-bone",
        "vehicle_structure": "single",
        "single_vehicle": "single",
        "vehicle_vehicle": "rear-end",
    }
    x = alias.get(x, x)
    if x in TYPE_SET:
        return x
    return None


def force_valid_type(primary=None, secondary=None, tertiary=None, fallback="single"):
    for cand in [primary, secondary, tertiary]:
        t = sanitize_type(cand)
        if t in TYPE_SET:
            return t
    return fallback


def normalize_window_result(r, frames, default_time):
    if not isinstance(r, dict):
        return {"time_sec": default_time, "score": 0.0, "description": "", "accident_type": "single"}
    try:
        t = float(r.get("time_sec", default_time))
    except Exception:
        t = default_time
    try:
        score = float(r.get("score", 0))
    except Exception:
        score = 0.0
    return {
        "time_sec": nearest_timestamp(t, frames),
        "score": score,
        "description": str(r.get("description", "")),
        "accident_type": force_valid_type(r.get("accident_type"), fallback="single"),
    }


def normalize_refine_result(r, frames, peak_fallback, fallback_type):
    if not isinstance(r, dict):
        return {
            "start_sec": max(0.0, peak_fallback - 0.5),
            "peak_sec": peak_fallback,
            "end_sec": peak_fallback + 1.0,
            "description": "",
            "accident_type": fallback_type,
            "confidence": 0.0,
        }
    try:
        start = float(r.get("start_sec", peak_fallback - 0.5))
    except Exception:
        start = peak_fallback - 0.5
    try:
        peak = float(r.get("peak_sec", peak_fallback))
    except Exception:
        peak = peak_fallback
    try:
        end = float(r.get("end_sec", peak_fallback + 1.0))
    except Exception:
        end = peak_fallback + 1.0
    try:
        confidence = float(r.get("confidence", 0))
    except Exception:
        confidence = 0.0
    return {
        "start_sec": start,
        "peak_sec": nearest_timestamp(peak, frames),
        "end_sec": end,
        "description": str(r.get("description", "")),
        "accident_type": force_valid_type(r.get("accident_type"), fallback=fallback_type),
        "confidence": confidence,
    }


def candidate_final_score(candidate):
    coarse = float(candidate.get("coarse_score", 0))
    mid = float(candidate.get("mid_score", 0))
    final = float(candidate.get("final_score", 0))
    bonus = 7.0
    if candidate.get("peak") is not None:
        bonus += 2.0
    return coarse * 0.25 + mid * 0.30 + final * 0.45 + bonus


# ─────────────────────────────────────────────
