#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CASCADE-VLM inference script.

Run accident detection on a single video or a directory of videos,
and update a submission CSV.

Usage:
    python scripts/run_inference.py \\
        --video-dir /path/to/videos \\
        --metadata-csv /path/to/metadata.csv \\
        --submission-template /path/to/template.csv \\
        --submission-out /path/to/output.csv \\
        --output-dir ./output

For a single video:
    python scripts/run_inference.py \\
        --video /path/to/single_video.mp4 \\
        --metadata-csv /path/to/metadata.csv \\
        --submission-template /path/to/template.csv \\
        --submission-out /path/to/output.csv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import cascade_vlm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_vlm.client import get_client, check_server
from cascade_vlm.config import VLLM_URL, MODEL, DEFAULT_VIDEO_DIR, DEFAULT_METADATA_CSV
from cascade_vlm.metadata import load_metadata
from cascade_vlm.pipeline import detect
from cascade_vlm.submission import (
    load_submission_template, update_submission_row,
    result_to_submission_fields, save_submission_atomic,
)
from cascade_vlm.visualization import render


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description="CASCADE-VLM: zero-shot CCTV accident detection."
    )
    ap.add_argument("--video", type=str,
                    help="Path to a single video file")
    ap.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR,
                    help="Directory containing video files")
    ap.add_argument("--metadata-csv", type=str, default=DEFAULT_METADATA_CSV,
                    help="Per-clip metadata CSV (with columns like road_geometry, weather, etc.)")
    ap.add_argument("--submission-template", type=str, required=True,
                    help="Submission CSV template")
    ap.add_argument("--submission-out", type=str, required=True,
                    help="Output submission CSV path")
    ap.add_argument("--output-dir", type=str, default="./output",
                    help="Output directory for per-video reports and visualizations")
    ap.add_argument("--server", type=str, default=VLLM_URL,
                    help="vLLM server URL")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Start index for resuming evaluation")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip video rendering (faster)")
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check server
    if not check_server(args.server):
        logger.error(f"Failed to connect to vLLM server: {args.server}")
        sys.exit(1)

    client = get_client(args.server)
    meta_df = load_metadata(args.metadata_csv)

    # Initialize submission CSV
    sub_df = load_submission_template(args.submission_template)
    save_submission_atomic(sub_df, args.submission_out)
    logger.info(f"submission template loaded: {args.submission_template}")
    logger.info(f"submission output initialized: {args.submission_out}")

    # Collect videos
    if args.video:
        vids = [args.video]
    else:
        vids = sorted(
            str(p) for p in Path(args.video_dir).iterdir()
            if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".wmv")
        )
        vids = vids[args.start_idx:]

    if not vids:
        logger.error(f"No videos found in: {args.video_dir}")
        sys.exit(1)

    logger.info(f"Found {len(vids)} videos. Model: {MODEL}")

    # Run inference per video
    for vp in vids:
        stem = Path(vp).stem
        logger.info(f"\n{'=' * 60}\nAnalyzing: {vp}\n{'=' * 60}")

        result = detect(client, vp, meta_df=meta_df)

        # Save JSON report
        report_path = os.path.join(args.output_dir, f"{stem}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        # Render visualization (optional)
        if not args.no_render:
            out_video = os.path.join(args.output_dir, f"{stem}_result.mp4")
            render(vp, result, out_video)
        else:
            out_video = None

        # Update submission
        video_rel_path = f"videos/{Path(vp).name}"
        sub_df = update_submission_row(sub_df, video_rel_path, result)
        save_submission_atomic(sub_df, args.submission_out)

        pred_row = result_to_submission_fields(video_rel_path, result)

        # Print summary
        print(f"\n🚨 {Path(vp).name}")
        print(f"   Accident type: {pred_row['type']}")
        print(f"   Impact time:   {pred_row['accident_time']:.6f}s")
        print(f"   Contact point: ({pred_row['center_x']:.6f}, {pred_row['center_y']:.6f})")
        if result.get("accident_point"):
            p = result["accident_point"]
            print(f"   Pixel coords:  ({p['x']}, {p['y']})")
        if out_video:
            print(f"   Video output:  {out_video}")
        print(f"   Report:        {report_path}")
        print(f"   Submission:    {args.submission_out}")

    print(f"\nDone.")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    print(f"Submission: {os.path.abspath(args.submission_out)}")


if __name__ == "__main__":
    main()
