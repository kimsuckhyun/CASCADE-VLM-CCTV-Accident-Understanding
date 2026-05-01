#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video IO, frame extraction, and spatial region generation.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import os
import cv2
import numpy as np

from .config import (
    COARSE_MIN_SIDE, COARSE_MAX_SIDE,
    SPATIAL_BASE_DIV, SPATIAL_STRIDE_RATIO,
)


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = total / fps if fps and fps > 0 else 0.0
    return {"fps": fps, "w": w, "h": h, "total": total, "dur": dur}


def sec_to_frame_idx(sec, fps, total_frames=None):
    idx = int(round(sec * fps))
    if total_frames is not None:
        idx = max(0, min(idx, total_frames - 1))
    return idx


def stamp_frame(frame, frame_idx, total_frames, timestamp):
    h, w = frame.shape[:2]
    label = f"[{frame_idx+1}/{total_frames}] {timestamp:.2f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.7, w / 1200.0))
    thickness = max(1, int(scale * 2.5))
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + pad * 2 + baseline), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, label, (pad, th + pad), font, scale, (0, 255, 255), thickness)
    return frame


def resize_keep_ratio_min_side(frame, target_min_side=520, max_side_cap=960):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    scale = target_min_side / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if max(new_h, new_w) > max_side_cap:
        cap_scale = max_side_cap / float(max(new_h, new_w))
        new_w = max(1, int(round(new_w * cap_scale)))
        new_h = max(1, int(round(new_h * cap_scale)))

    if new_w == w and new_h == h:
        return frame.copy()
    return cv2.resize(frame, (new_w, new_h))


def extract_frames_uniform(path, fps, start=0.0, end=None, max_n=32, min_side=520, max_side=960, do_stamp=True):
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / vfps if vfps > 0 else 0.0

    if end is None:
        end = dur
    end = min(end, dur)

    gap = 1.0 / fps
    out = []
    sec = start

    raw_list = []
    while sec <= end + 1e-9:
        if len(raw_list) >= max_n:
            break
        idx = sec_to_frame_idx(sec, vfps, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok or fr is None:
            sec += gap
            continue
        raw_list.append({"frame": fr, "sec": round(float(sec), 3)})
        sec += gap
    cap.release()

    total_frames = len(raw_list)
    for i, item in enumerate(raw_list):
        fr_small = resize_keep_ratio_min_side(item["frame"], min_side, max_side)
        if do_stamp:
            fr_small = stamp_frame(fr_small, i, total_frames, item["sec"])

        ok, buf = cv2.imencode(".jpg", fr_small, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue

        out.append({
            "frame": item["frame"],
            "display_frame": fr_small,
            "sec": item["sec"],
            "b64": base64.b64encode(buf).decode(),
        })

    logger.info(f"  프레임 {len(out)}장 추출 ({start:.2f}s~{end:.2f}s, sample_fps={fps}, min_side={min_side}, max_side={max_side})")
    return out


def generate_dynamic_time_segments(duration):
    if duration <= 15.0:
        return {"fps": 4.0, "segments": [(0.0, round(duration, 3))], "segment_len": duration, "stride": duration}
    if duration <= 30.0:
        seg_len = duration / 2.0
        stride = seg_len / 2.0
    else:
        seg_len = duration / 3.0
        stride = seg_len / 2.0

    segments = []
    s = 0.0
    while s < duration - 1e-9:
        e = min(duration, s + seg_len)
        segments.append((round(s, 3), round(e, 3)))
        if e >= duration:
            break
        s += stride
    return {"fps": 4.0, "segments": segments, "segment_len": seg_len, "stride": stride}


def get_dense_coarse_frame_budget(seg_start, seg_end, fps):
    """
    coarse stage에서 time segment 전체를 그대로 보도록 프레임 budget 계산.
    예: 15초 * 8fps = 120장
    """
    dur = max(0.0, float(seg_end) - float(seg_start))
    return max(1, int(round(dur * float(fps))))


def generate_overlap_spatial_regions(video_w, video_h, base_div=3, stride_ratio=0.5):
    crop_w = int(round(video_w / base_div))
    crop_h = int(round(video_h / base_div))
    crop_w = max(32, min(video_w, crop_w))
    crop_h = max(32, min(video_h, crop_h))

    stride_x = max(1, int(round(crop_w * stride_ratio)))
    stride_y = max(1, int(round(crop_h * stride_ratio)))

    x_starts = list(range(0, max(1, video_w - crop_w + 1), stride_x))
    y_starts = list(range(0, max(1, video_h - crop_h + 1), stride_y))

    if x_starts[-1] != video_w - crop_w:
        x_starts.append(max(0, video_w - crop_w))
    if y_starts[-1] != video_h - crop_h:
        y_starts.append(max(0, video_h - crop_h))

    x_starts = sorted(set(x_starts))
    y_starts = sorted(set(y_starts))

    regions = []
    idx = 1
    for yi, y1 in enumerate(y_starts):
        for xi, x1 in enumerate(x_starts):
            x2 = min(video_w, x1 + crop_w)
            y2 = min(video_h, y1 + crop_h)
            regions.append({
                "region_id": idx,
                "grid_pos": f"row{yi+1}_col{xi+1}",
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "w": int(x2 - x1), "h": int(y2 - y1),
            })
            idx += 1
    return regions


def draw_region_box(frame, region, label=None, color=(0, 255, 0), thickness=3):
    img = frame.copy()
    x1, y1, x2, y2 = int(region["x1"]), int(region["y1"]), int(region["x2"]), int(region["y2"])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        by1 = max(0, y1 - th - 10)
        cv2.rectangle(img, (x1, by1), (x1 + tw + 10, by1 + th + 10), color, -1)
        cv2.putText(img, label, (x1 + 5, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def make_side_by_side_context_image(full_frame_raw, region, timestamp, frame_idx, total_frames,
                                    crop_min_side=520, crop_max_side=960,
                                    context_min_side=360, context_max_side=700):
    full_marked = draw_region_box(full_frame_raw, region, label=f"LOCAL region #{region['region_id']}")
    full_marked = resize_keep_ratio_min_side(full_marked, context_min_side, context_max_side)

    x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
    crop = full_frame_raw[y1:y2, x1:x2]
    crop = resize_keep_ratio_min_side(crop, crop_min_side, crop_max_side)

    h_full, w_full = full_marked.shape[:2]
    h_crop, w_crop = crop.shape[:2]
    target_h = max(h_full, h_crop)

    def fit_to_h(img, target_h):
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / float(h)
        return cv2.resize(img, (max(1, int(round(w * scale))), target_h))

    full_marked = fit_to_h(full_marked, target_h)
    crop = fit_to_h(crop, target_h)

    gap = 16
    canvas = np.zeros((target_h + 60, full_marked.shape[1] + crop.shape[1] + gap, 3), dtype=np.uint8)
    canvas[:target_h, :full_marked.shape[1]] = full_marked
    canvas[:target_h, full_marked.shape[1] + gap:] = crop

    cv2.putText(canvas, "FULL CONTEXT (region highlighted)", (8, target_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(canvas, "LOCAL CROP", (full_marked.shape[1] + gap + 8, target_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    canvas = stamp_frame(canvas, frame_idx, total_frames, timestamp)
    return canvas


def region_to_text(region, video_w, video_h):
    cx = (region["x1"] + region["x2"]) / 2.0
    cy = (region["y1"] + region["y2"]) / 2.0

    if cx < video_w / 3:
        hx = "left"
    elif cx < 2 * video_w / 3:
        hx = "center"
    else:
        hx = "right"

    if cy < video_h / 3:
        hy = "top"
    elif cy < 2 * video_h / 3:
        hy = "middle"
    else:
        hy = "bottom"

    return (
        f"region #{region['region_id']} ({region['grid_pos']}), approx {hy}-{hx}, "
        f"pixel box=({region['x1']},{region['y1']})-({region['x2']},{region['y2']}), "
        f"size={region['w']}x{region['h']}"
    )


# ─────────────────────────────────────────────
