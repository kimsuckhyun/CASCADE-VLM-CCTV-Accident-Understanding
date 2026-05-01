#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Recursive 3x3 Grid-based Point Localization.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import re
import base64
import logging
import cv2
import numpy as np

from .config import GRID_ROWS, GRID_COLS, GRID_ROUNDS

logger = logging.getLogger(__name__)
from .parsing import _extract_response_text, force_valid_type, safe_json_loads
from .video_utils import resize_keep_ratio_min_side, sec_to_frame_idx, stamp_frame


GRID_SYSTEM_PROMPT = f"""You are an expert CCTV vehicle accident analyst specializing in spatial localization.

You are given CCTV frame(s) with a numbered grid overlay (green lines and numbers).
Your task is to identify which numbered grid cell contains the FIRST physical contact point of the accident.
Frames have "[N/Total] T.TTs" labels showing chronological order.

{VEHICLE_ACCIDENT_DEFINITION}
{LOW_VISIBILITY_RULES}
{LENS_OBSTRUCTION_RULES}
{DISTANT_SMALL_ACCIDENT_RULES}
{FALSE_POSITIVE_RULES}

Output ONLY one valid JSON object. No reasoning, no markdown, no text before or after JSON.
"""


def draw_grid_on_frame(frame, rows=3, cols=3):
    img = frame.copy()
    h, w = img.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    for i in range(1, cols):
        x = int(i * cell_w)
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 3)
    for j in range(1, rows):
        y = int(j * cell_h)
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 3)

    num = 1
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            font_scale = max(0.8, min(2.5, min(cell_w, cell_h) / 120.0))
            thickness = max(2, int(font_scale * 2))
            text = str(num)
            (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pad = 8
            cv2.rectangle(img, (cx - tw // 2 - pad, cy - th_t // 2 - pad),
                          (cx + tw // 2 + pad, cy + th_t // 2 + pad), (0, 0, 0), -1)
            cv2.rectangle(img, (cx - tw // 2 - pad, cy - th_t // 2 - pad),
                          (cx + tw // 2 + pad, cy + th_t // 2 + pad), (0, 255, 0), 2)
            cv2.putText(img, text, (cx - tw // 2, cy + th_t // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            num += 1
    return img


def frame_to_b64(frame, quality=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else None


def ask_grid(client, b64_images, prompt, max_tok=1200):
    content = []
    for b in b64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": prompt + "\nReturn ONLY one raw JSON object. Your first character must be '{' and your last character must be '}'."})
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": GRID_SYSTEM_PROMPT}, {"role": "user", "content": content}],
        max_tokens=max_tok,
        temperature=0.0,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return _extract_response_text(r)


def parse_grid_choice(raw_text):
    r = safe_json_loads(raw_text)
    cell = r.get("cell")
    if cell is not None:
        try:
            cell = int(cell)
            if 1 <= cell <= GRID_ROWS * GRID_COLS:
                return cell, r
        except Exception:
            pass
    nums = re.findall(r"\b([1-9])\b", str(raw_text))
    if nums:
        cell = int(nums[0])
        if 1 <= cell <= GRID_ROWS * GRID_COLS:
            return cell, r
    return 5, r


def grid_localize_accident_point(client, vpath, result, video_w, video_h, dur, meta_block, init_region=None):
    peak = result["peak"]
    acc_type = force_valid_type(result.get("accident_type"), fallback="single")
    offsets = [-1.0, -0.5, -0.15, 0.0, 0.15, 0.5]
    times = [max(0.0, min(dur, peak + dt)) for dt in offsets]

    cap = cv2.VideoCapture(vpath)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_frames = []
    for sec in times:
        idx = sec_to_frame_idx(sec, vfps, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok and fr is not None:
            raw_frames.append({"frame": fr, "sec": sec})
    cap.release()

    if not raw_frames:
        result["accident_point"] = {"x": video_w // 2, "y": video_h // 2}
        return result

    if init_region is None:
        current_region = {"x1": 0, "y1": 0, "x2": video_w, "y2": video_h}
    else:
        current_region = {
            "x1": int(init_region["x1"]), "y1": int(init_region["y1"]),
            "x2": int(init_region["x2"]), "y2": int(init_region["y2"])
        }

    for round_idx in range(GRID_ROUNDS):
        rw = current_region["x2"] - current_region["x1"]
        rh = current_region["y2"] - current_region["y1"]
        if rw < 20 or rh < 20:
            break

        grid_b64_list = []
        total_grid_frames = len(raw_frames)
        for fi, rf in enumerate(raw_frames):
            cropped = rf["frame"][current_region["y1"]:current_region["y2"], current_region["x1"]:current_region["x2"]]
            cropped = resize_keep_ratio_min_side(cropped, 400, 900)
            cropped = stamp_frame(cropped, fi, total_grid_frames, rf["sec"])
            grid_img = draw_grid_on_frame(cropped, GRID_ROWS, GRID_COLS)
            b64 = frame_to_b64(grid_img, quality=88)
            if b64:
                grid_b64_list.append(b64)

        if not grid_b64_list:
            break

        round_context = "a zoomed-in crop" if round_idx > 0 else "the candidate region crop"

        prompt = f"""
{meta_block}

You are shown {len(grid_b64_list)} CCTV frame(s) of {round_context}.
Frames are in CHRONOLOGICAL ORDER.
Each frame has a 3x3 numbered grid overlay.

Frames span from before to after the estimated accident moment ({peak:.2f}s).
The accident type is estimated as: {acc_type}

{VEHICLE_ACCIDENT_DEFINITION}
{ACCIDENT_TYPE_5_DEFINITION}
{LOW_VISIBILITY_RULES}
{LENS_OBSTRUCTION_RULES}
{DISTANT_SMALL_ACCIDENT_RULES}
{FALSE_POSITIVE_RULES}

Task: Which numbered cell (1-9) contains the FIRST physical contact point of the accident?

Respond ONLY in JSON:
{{
  "cell": <integer 1-9>,
  "reason": "<brief explanation>"
}}
"""
        raw = ask_grid(client, grid_b64_list, prompt)
        cell_num, r_json = parse_grid_choice(raw)
        logger.info(f"  [Grid Round {round_idx+1}] → cell {cell_num}")

        row_idx = (cell_num - 1) // GRID_COLS
        col_idx = (cell_num - 1) % GRID_COLS

        cell_x1 = current_region["x1"] + int(rw * col_idx / GRID_COLS)
        cell_y1 = current_region["y1"] + int(rh * row_idx / GRID_ROWS)
        cell_x2 = current_region["x1"] + int(rw * (col_idx + 1) / GRID_COLS)
        cell_y2 = current_region["y1"] + int(rh * (row_idx + 1) / GRID_ROWS)

        pad_w = int(rw * 0.05)
        pad_h = int(rh * 0.05)
        current_region = {
            "x1": max(0, cell_x1 - pad_w),
            "y1": max(0, cell_y1 - pad_h),
            "x2": min(video_w, cell_x2 + pad_w),
            "y2": min(video_h, cell_y2 + pad_h),
        }

    cx = (current_region["x1"] + current_region["x2"]) // 2
    cy = (current_region["y1"] + current_region["y2"]) // 2
    result["accident_point"] = {"x": max(0, min(video_w - 1, cx)), "y": max(0, min(video_h - 1, cy))}
    return result


# ─────────────────────────────────────────────
