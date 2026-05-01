#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Local Scanning and Candidate Concatenation.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import logging

from .config import COARSE_TOP_K, CLUSTER_SCORE_THRESHOLD, CLUSTER_IOU_THRESHOLD

logger = logging.getLogger(__name__)
from .video_utils import make_side_by_side_context_image


def build_region_context_frames(frames, region):
    out = []
    total_frames = len(frames)
    for i, fd in enumerate(frames):
        combo = make_side_by_side_context_image(fd["frame"], region, fd["sec"], i, total_frames,
                                                crop_min_side=COARSE_MIN_SIDE, crop_max_side=COARSE_MAX_SIDE)
        ok, buf = cv2.imencode(".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            continue
        out.append({"sec": fd["sec"], "b64": base64.b64encode(buf).decode()})
    return out


def coarse_prompt_for_region(meta_block, actual_start, actual_end, fps, nframes, region_text):
    return f"""
{meta_block}

You are given sequential CCTV images for ONE local spatial candidate region of a time segment.

Each image contains:
- LEFT: FULL FRAME with the local region highlighted
- RIGHT: the corresponding LOCAL CROP enlarged

The full frame is ONLY for context.
Your scoring must focus on whether the FIRST physical accident happens inside the LOCAL region.

Local region info:
- {region_text}

{VEHICLE_ACCIDENT_DEFINITION}
{ACCIDENT_TYPE_5_DEFINITION}
{LOW_VISIBILITY_RULES}
{LENS_OBSTRUCTION_RULES}
{DISTANT_SMALL_ACCIDENT_RULES}
{FALSE_POSITIVE_RULES}
{SCORE_CALIBRATION}

Segment info:
- Actual provided frame span: {actual_start:.2f}s to {actual_end:.2f}s
- Sample rate: {fps:.2f} fps
- Number of frames: {nframes}

Task:
1. Determine if the FIRST physical accident is visible INSIDE the LOCAL region.
2. If yes, estimate the best timestamp and type.
3. Assign a score.

Important:
- time_sec should be one of the provided timestamps.
- If the accident is visible elsewhere in the full frame but NOT inside the LOCAL region, score LOW.
- A high score means THIS local region contains the first physical contact.
- Do NOT output unknown.

Respond ONLY in JSON:
{{
  "time_sec": <float>,
  "score": <0-100>,
  "accident_type": "single" or "rear-end" or "t-bone" or "sideswipe" or "head-on",
  "description": "<brief reason focused on this local region>"
}}
"""


def select_top_candidates_zscore(candidates, top_k=COARSE_TOP_K):
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return sorted(candidates, key=lambda x: float(x.get("score", 0)), reverse=True)

    scores = np.array([float(c.get("score", 0)) for c in candidates], dtype=float)
    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))
    if std_s > 1e-6:
        for c, s in zip(candidates, scores):
            c["z_score"] = float((s - mean_s) / std_s)
    else:
        for c in candidates:
            c["z_score"] = 0.0

    ranked = sorted(candidates, key=lambda x: (x.get("z_score", 0), float(x.get("score", 0))), reverse=True)
    logger.info(f"  coarse score stats: mean={mean_s:.2f}, std={std_s:.2f}")
    return ranked[:top_k]



def region_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter) / float(max(1, union))


def union_regions(regions, video_w, video_h):
    x1 = min(r["x1"] for r in regions)
    y1 = min(r["y1"] for r in regions)
    x2 = max(r["x2"] for r in regions)
    y2 = max(r["y2"] for r in regions)
    return {
        "x1": max(0, int(x1)),
        "y1": max(0, int(y1)),
        "x2": min(video_w, int(x2)),
        "y2": min(video_h, int(y2)),
        "w": min(video_w, int(x2)) - max(0, int(x1)),
        "h": min(video_h, int(y2)) - max(0, int(y1)),
        "region_id": -1,
        "grid_pos": "cluster_union",
    }


def cluster_spatial_candidates(candidates, video_w, video_h, score_threshold=70.0, iou_threshold=0.15):
    """
    겹치는 high-score spatial candidates를 하나의 사고 cluster로 묶는다.
    같은 segment 내에서만 클러스터링한다.
    """
    strong = [c for c in candidates if float(c.get("score", 0)) >= score_threshold]
    if not strong:
        return []

    strong = sorted(strong, key=lambda x: (int(x.get("segment_idx", 0)), float(x.get("time_sec", 0)), -float(x.get("score", 0))))
    used = [False] * len(strong)
    clusters = []

    for i, cand in enumerate(strong):
        if used[i]:
            continue
        used[i] = True
        cluster = [cand]

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(strong):
                if used[j]:
                    continue
                if int(other.get("segment_idx", -1)) != int(cand.get("segment_idx", -1)):
                    continue
                for member in cluster:
                    if region_iou(member["region"], other["region"]) >= iou_threshold:
                        used[j] = True
                        cluster.append(other)
                        changed = True
                        break
        # 대표 후보: earliest-impact 우선, 같은 시간이면 더 높은 score
        rep = sorted(cluster, key=lambda x: (float(x.get("time_sec", 0)), -float(x.get("score", 0))))[0]
        best_score = max(float(x.get("score", 0)) for x in cluster)
        union_region = union_regions([x["region"] for x in cluster], video_w, video_h)

        rep2 = dict(rep)
        rep2["cluster_size"] = len(cluster)
        rep2["cluster_members"] = cluster
        rep2["cluster_best_score"] = best_score
        rep2["region_union"] = union_region
        rep2["score"] = best_score  # cluster 대표 점수는 cluster 최고점 사용
        clusters.append(rep2)
    return clusters


def select_cluster_representatives(candidates, video_w, video_h, top_k=COARSE_TOP_K,
                                   score_threshold=70.0, iou_threshold=0.15):
    """
    1) 겹치는 high-score region들을 cluster로 묶음
    2) 각 cluster에서 earliest-impact 후보를 대표로 선택
    3) cluster 최고점 기준으로 top-k 선택
    """
    clusters = cluster_spatial_candidates(
        candidates, video_w, video_h,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    if not clusters:
        # fallback: 기존 zscore 상위 후보
        return select_top_candidates_zscore(candidates, top_k=top_k), []

    ranked = sorted(
        clusters,
        key=lambda x: (-float(x.get("cluster_best_score", x.get("score", 0))),
                       float(x.get("time_sec", 0)))
    )
    return ranked[:top_k], clusters

# ─────────────────────────────────────────────
