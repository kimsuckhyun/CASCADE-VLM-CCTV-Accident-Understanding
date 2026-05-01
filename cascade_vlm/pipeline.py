#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline: orchestrates Stages 1-3 to produce final predictions.

Part of CASCADE-VLM (CVPR 2026 ACCIDENT Challenge submission).
"""

import logging

logger = logging.getLogger(__name__)
from .metadata import build_metadata_prompt_block, clean_meta_value, get_metadata_for_video, meta_number
from .parsing import (
    ask_frames, candidate_final_score, clamp_time, force_valid_type,
    normalize_window_result, safe_json_loads,
)
from .stage1_coarse import (
    build_region_context_frames, coarse_prompt_for_region,
    select_cluster_representatives,
)
from .stage2_refine import refine_candidate, resolve_type_with_prior, verify_accident_type
from .stage3_grid import grid_localize_accident_point
from .video_utils import (
    extract_frames_uniform, generate_dynamic_time_segments,
    generate_overlap_spatial_regions, get_dense_coarse_frame_budget,
    region_to_text, video_info,
)


def detect(client, vpath, meta_df=None):
    vi = video_info(vpath)
    dur, w, h = vi["dur"], vi["w"], vi["h"]
    logger.info(f"영상: {dur:.2f}s, {w}x{h}, {vi['fps']:.2f}fps")

    meta = get_metadata_for_video(meta_df, vpath)
    meta_block = build_metadata_prompt_block(meta)

    time_plan = generate_dynamic_time_segments(dur)
    scan_fps = float(time_plan["fps"])
    segments = time_plan["segments"]
    segment_len = time_plan["segment_len"]
    stride_sec = time_plan["stride"]

    spatial_regions = generate_overlap_spatial_regions(w, h, base_div=SPATIAL_BASE_DIV, stride_ratio=SPATIAL_STRIDE_RATIO)
    logger.info(f"[1/5] coarse spatial-temporal scan: time_segments={len(segments)}, spatial_regions={len(spatial_regions)}")

    coarse_candidates = []
    for si, (ws, we) in enumerate(segments, 1):
        logger.info(f"  segment {si}/{len(segments)} : {ws:.2f}s ~ {we:.2f}s")
        coarse_max_n = get_dense_coarse_frame_budget(ws, we, scan_fps)
        frames = extract_frames_uniform(vpath, fps=scan_fps, start=ws, end=we, max_n=coarse_max_n,
                                        min_side=COARSE_MIN_SIDE, max_side=COARSE_MAX_SIDE, do_stamp=False)
        if not frames:
            continue

        actual_start = frames[0]["sec"]
        actual_end = frames[-1]["sec"]

        for region in spatial_regions:
            region_frames = build_region_context_frames(frames, region)
            if not region_frames:
                continue

            prompt = coarse_prompt_for_region(
                meta_block=meta_block,
                actual_start=actual_start,
                actual_end=actual_end,
                fps=scan_fps,
                nframes=len(region_frames),
                region_text=region_to_text(region, w, h),
            )
            raw = ask_frames(client, region_frames, prompt, max_tok=2000, max_images=len(region_frames))
            r = normalize_window_result(safe_json_loads(raw), region_frames, default_time=(actual_start + actual_end) / 2.0)
            r["_segment_start"] = ws
            r["_segment_end"] = we
            r["_actual_start"] = actual_start
            r["_actual_end"] = actual_end
            r["region"] = region
            r["segment_idx"] = si
            coarse_candidates.append(r)
            logger.info(f"    region {region['region_id']:02d} ({region['grid_pos']}) → score={r['score']:.1f}, time={r['time_sec']}, type={r['accident_type']}")

    top_windows, coarse_clusters = select_cluster_representatives(
        coarse_candidates, w, h, top_k=COARSE_TOP_K,
        score_threshold=70.0, iou_threshold=0.15
    )
    logger.info(f"[2/5] clustered top coarse candidates = {len(top_windows)} / clusters={len(coarse_clusters)}")

    candidates = []
    if top_windows:
        for i, cw in enumerate(top_windows, 1):
            rough = clamp_time(cw.get("time_sec"), dur)
            if rough is None:
                rough = (cw["_actual_start"] + cw["_actual_end"]) / 2.0
            cand = refine_candidate(client, vpath, dur, rough, cw, meta_block, w, h)
            cand["candidate_rank"] = i
            cand["candidate_score"] = candidate_final_score(cand)
            candidates.append(cand)
            logger.info(f"  [candidate {i}] score={cand['candidate_score']:.2f}, type={cand['accident_type']}")
    else:
        fallback_region = {"x1": 0, "y1": 0, "x2": w, "y2": h, "w": w, "h": h, "region_id": -1, "grid_pos": "fallback"}
        rough = dur / 2.0
        fake_cw = {
            "score": 0,
            "description": "fallback candidate",
            "accident_type": "single",
            "_actual_start": max(0.0, rough - 3.0),
            "_actual_end": min(dur, rough + 3.0),
            "region": fallback_region,
        }
        cand = refine_candidate(client, vpath, dur, rough, fake_cw, meta_block, w, h)
        cand["candidate_rank"] = 1
        cand["candidate_score"] = candidate_final_score(cand)
        candidates.append(cand)

    best_candidate = max(candidates, key=lambda x: x["candidate_score"])

    logger.info("[3/5] Type 검증")
    current_type = force_valid_type(best_candidate.get("accident_type"), fallback="single")
    verified_type = verify_accident_type(
        client, vpath, best_candidate["peak"], dur, current_type, meta_block, meta,
        region=best_candidate.get("refine_region")
    )
    type_chain = best_candidate.get("_type_chain", [current_type, current_type, current_type])
    final_type = resolve_type_with_prior(type_chain, verified_type, meta)
    best_candidate["accident_type"] = final_type

    result = {
        "detected": True,
        "metadata_used": {
            "scene_layout": clean_meta_value(meta.get("scene_layout")) if meta else "unknown",
            "weather": clean_meta_value(meta.get("weather")) if meta else "unknown",
            "day_time": clean_meta_value(meta.get("day_time")) if meta else "unknown",
            "quality": clean_meta_value(meta.get("quality")) if meta else "unknown",
            "region": clean_meta_value(meta.get("region")) if meta else "unknown",
            "meta_duration": meta_number(meta, ["duration", "video_duration"], default="unknown", cast=float) if meta else "unknown",
            "meta_height": meta_number(meta, ["height", "h"], default="unknown", cast=int) if meta else "unknown",
            "meta_width": meta_number(meta, ["width", "w"], default="unknown", cast=int) if meta else "unknown",
            "meta_no_frames": meta_number(meta, ["no_frames", "num_frames"], default="unknown", cast=int) if meta else "unknown",
        },
        "accident_type": final_type,
        "road_context": "",
        "start": best_candidate["start"],
        "peak": best_candidate["peak"],
        "end": best_candidate["end"],
        "accident_point": None,
        "description": best_candidate.get("description", ""),
        "video_info": vi,
        "debug": {
            "scan_fps": scan_fps,
            "segment_len": segment_len,
            "stride_sec": stride_sec,
            "segments": segments,
            "spatial_regions_count": len(spatial_regions),
            "top_windows": top_windows,
            "coarse_clusters": coarse_clusters,
            "candidates": candidates,
            "best_candidate": best_candidate,
            "type_chain": type_chain,
            "verified_type": verified_type,
            "final_type": final_type,
        },
    }

    logger.info("[4/5] Grid 기반 사고 점(point) localization")
    result = grid_localize_accident_point(
        client, vpath, result, w, h, dur, meta_block, init_region=best_candidate.get("refine_region")
    )

    if result.get("accident_point") is None:
        result["accident_point"] = {"x": w // 2, "y": h // 2}
    return result


# ─────────────────────────────────────────────
