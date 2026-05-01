#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result visualization on input video (overlay accident point + type).

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import os
import subprocess
import shutil
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def draw_accident_point(frame, point, label="ACCIDENT", color=(0, 0, 255)):
    if not point:
        return frame
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, int(point["x"])))
    y = max(0, min(h - 1, int(point["y"])))
    cv2.circle(frame, (x, y), 10, color, -1)
    cv2.circle(frame, (x, y), 22, color, 2)
    cross_len = 28
    cv2.line(frame, (max(0, x - cross_len), y), (min(w - 1, x + cross_len), y), color, 2)
    cv2.line(frame, (x, max(0, y - cross_len)), (x, min(h - 1, y + cross_len)), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bx1 = min(max(0, x + 12), max(0, w - tw - 12))
    by1 = max(0, y - 14 - th)
    cv2.rectangle(frame, (bx1, by1), (bx1 + tw + 10, by1 + th + 10), color, -1)
    cv2.putText(frame, label, (bx1 + 5, by1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def transcode_to_h264(src_path, dst_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(src_path, dst_path)
        return
    cmd = [ffmpeg, "-y", "-i", src_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(src_path):
            os.remove(src_path)
    except subprocess.CalledProcessError:
        shutil.move(src_path, dst_path)


def render(vpath, result, outpath):
    if not result.get("detected"):
        return
    temp_out = str(Path(outpath).with_suffix(".tmp.mp4"))
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if fps > 0 else 0.0
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    pk, st, en = result["peak"], result["start"], result["end"]
    point = result.get("accident_point")
    acc_type = result.get("accident_type", "single")
    RED, WHT, YEL = (0, 0, 255), (255, 255, 255), (0, 255, 255)
    GREEN = (0, 255, 0)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sec = fi / fps if fps > 0 else 0.0
        at_pk = abs(sec - pk) < 0.7
        near = abs(sec - pk) < 2.0
        zone = (st - 1.5) <= sec <= (en + 1.5)

        if at_pk:
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, h), RED, -1)
            frame = cv2.addWeighted(ov, 0.12, frame, 0.88, 0)
        if zone and point:
            if near:
                ov = frame.copy()
                cv2.circle(ov, (int(point["x"]), int(point["y"])), 34 if at_pk else 26, RED, -1)
                frame = cv2.addWeighted(ov, 0.20 if at_pk else 0.10, frame, 0.80 if at_pk else 0.90, 0)
            frame = draw_accident_point(frame, point, label=f"ACCIDENT @ {pk:.2f}s", color=RED)

        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (w, 85), RED if zone else (30, 30, 30), -1)
        frame = cv2.addWeighted(hud, 0.6 if zone else 0.4, frame, 0.4 if zone else 0.6, 0)
        cv2.putText(frame, f"playback: {sec:.2f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHT, 2)
        cv2.putText(frame, f"impact: {pk:.2f}s", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YEL, 2)
        cv2.putText(frame, f"type: {acc_type}", (400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        if point:
            cv2.putText(frame, f"point: ({point['x']}, {point['y']})", (700, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHT, 2)
        if at_pk:
            cv2.putText(frame, "!! IMPACT !!", (w // 2 - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, YEL, 3)
        elif zone:
            txt = "PRE-CRASH" if sec < pk else "POST-CRASH"
            cv2.putText(frame, txt, (w // 2 - 70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHT, 2)

        tl_y, mg = h - 35, 30
        cv2.rectangle(frame, (mg, tl_y), (w - mg, tl_y + 20), (50, 50, 50), -1)
        if dur > 0:
            sx = int(mg + (w - 2 * mg) * (st / dur))
            ex = int(mg + (w - 2 * mg) * (min(en, dur) / dur))
            cx_tl = int(mg + (w - 2 * mg) * (min(sec, dur) / dur))
            px = int(mg + (w - 2 * mg) * (min(pk, dur) / dur))
            cv2.rectangle(frame, (sx, tl_y), (ex, tl_y + 20), RED, -1)
            cv2.line(frame, (cx_tl, tl_y - 3), (cx_tl, tl_y + 23), WHT, 2)
            cv2.line(frame, (px, tl_y - 6), (px, tl_y + 26), YEL, 3)
        if zone:
            b = 5 if at_pk else 3
            cv2.rectangle(frame, (b, b), (w - b, h - b), RED, b)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()
    transcode_to_h264(temp_out, outpath)


# ─────────────────────────────────────────────
