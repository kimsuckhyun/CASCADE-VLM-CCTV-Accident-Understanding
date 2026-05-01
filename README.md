# CASCADE-VLM

Official implementation of **"CASCADE-VLM: A Zero-Shot Cascade Pipeline for CCTV Accident Understanding"** (CVPR 2026 ACCIDENT Challenge submission).

## 🏆 Challenge Results

**[ACCIDENT @ CVPR 2026 Challenge](https://www.kaggle.com/competitions/accident/overview)** (Zero-shot benchmark):

| Leaderboard | Rank | ACCS |
|---|---|---|
| 🥇 **Public** | **1st** | **0.55478** |
| 🥈 **Private** | **2nd** | **0.56293** |

Achieved **without any training data** using a frozen Qwen3.5-27B model.

---

CASCADE-VLM is a zero-shot cascade pipeline that operates a frozen vision-language model (VLM) at three progressively finer spatio-temporal resolutions to predict (1) when an impact occurs, (2) where the first contact happens, and (3) which collision type is involved in CCTV footage — without using any real CCTV training labels.

## Key Features

- **Zero-shot**: No real CCTV labels used for training, prompt tuning, or per-video correction
- **Frozen VLM**: Single Qwen3.5-27B model used throughout the pipeline
- **Cascaded pipeline**: 3 stages (coarse scanning → temporal refinement → spatial localization)
- **Scene-conditioned prompts**: CARLA-derived scene priors injected into type verification only
- **~35 VLM calls per video**

## Detailed Results

On the full ACCIDENT zero-shot split (2,027 real CCTV clips):

| Method | T | S | C | ACCS |
|---|---|---|---|---|
| Molmo-7B (single-frame) | 0.343 | 0.488 | 0.293 | 0.358 |
| Qwen3.5-27B (single-frame) | 0.382 | 0.491 | 0.509 | 0.453 |
| CASCADE-VLM (Coarse) | 0.566 | 0.341 | 0.546 | 0.459 |
| CASCADE-VLM (Full) | 0.570 | 0.556 | 0.567 | 0.564 |
| **CASCADE-VLM (Full+Scene)** | **0.568** | **0.555** | **0.595** | **0.572** |

## Installation

```bash
git clone https://github.com/kimsuckhyun/CASCADE-VLM-CCTV-Accident-Understanding.git
cd CASCADE-VLM-CCTV-Accident-Understanding
pip install -r requirements.txt
```

## Setup vLLM Server

CASCADE-VLM requires a running vLLM server with Qwen3.5-27B:

```bash
bash scripts/start_vllm.sh
```

Or manually:

```bash
vllm serve Qwen/Qwen3.5-27B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --enforce-eager
```

## Usage

### Run inference on a dataset

```bash
python scripts/run_inference.py \
    --video-dir /path/to/cctv/videos \
    --metadata-csv /path/to/metadata.csv \
    --submission-template /path/to/submission_template.csv \
    --submission-out /path/to/output_submission.csv
```

### Run on a single video (programmatic)

```python
from cascade_vlm.client import get_client
from cascade_vlm.pipeline import detect
from cascade_vlm.metadata import load_metadata

client = get_client("http://localhost:8000/v1")
meta_df = load_metadata("metadata.csv")
result = detect(client, "path/to/video.mp4", meta_df=meta_df)

print(result)
# {
#   'predicted_accident_type': 't-bone',
#   'predicted_start_sec': 5.73,
#   'predicted_peak_sec': 6.808,
#   'predicted_end_sec': 8.23,
#   'predicted_point': (0.56, 0.50),
#   ...
# }
```

## Pipeline Architecture

```
Input Video (D seconds, W×H)
    │
    ▼
[Stage 1] Local Scanning & Candidate Concatenation
    - Partition into temporal segments (1/2/3 by duration)
    - 3×3 grid of overlapping local regions per segment
    - VLM scores each (segment, region) pair in [0, 100]
    - Concatenate adjacent high-scoring regions (score > 70)
    - Output: top K=3 merged candidates
    │
    ▼
[Stage 2] Cascaded Temporal Refinement
    - Mid pass: 8 fps within ±1.9s of coarse peak
    - Fine pass: 12 fps around mid-stage peak
    - Weighted combination of coarse, mid, fine scores
    - Output: refined peak time t̂
    │
    ▼
[Stage 3] Type Verification & Point Localization
    - Scene-conditioned type verification (7 frames around t̂)
    - Recursive 3×3 grid cell selection (3 zoom rounds)
    - Output: (t̂, x̂, ŷ, ĉ)
```

## Code Structure

```
cascade_vlm/
├── config.py            # Constants and hyperparameters
├── client.py            # VLM server connection
├── metadata.py          # Per-clip metadata + scene-conditioned prompts
├── video_utils.py       # Video IO, frame extraction, region generation
├── prompts.py           # All VLM prompt templates
├── parsing.py           # JSON parsing and result normalization
├── stage1_coarse.py     # Stage 1: Local scanning + candidate concatenation
├── stage2_refine.py     # Stage 2: Temporal refinement + type verification
├── stage3_grid.py       # Stage 3: Recursive grid-based point localization
├── pipeline.py          # Main detect() function (orchestrates all stages)
├── visualization.py     # Result visualization on video
└── submission.py        # Submission CSV update utilities
```

## Citation

```bibtex
@inproceedings{cascadevlm2026,
    title={CASCADE-VLM: A Zero-Shot Cascade Pipeline for CCTV Accident Understanding},
    author={Anonymous},
    booktitle={CVPR},
    year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [ACCIDENT @ CVPR 2026 Challenge](https://www.kaggle.com/competitions/accident/overview) organizers
- Qwen3.5-27B by Alibaba
- vLLM project for efficient inference