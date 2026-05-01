#!/bin/bash
# Start vLLM server with Qwen3.5-27B for CASCADE-VLM inference.

set -e

MODEL="Qwen/Qwen3.5-27B"
PORT=8000
TENSOR_PARALLEL=1
MAX_MODEL_LEN=16384

echo "Starting vLLM server..."
echo "  Model:           $MODEL"
echo "  Port:            $PORT"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo "  Max model len:   $MAX_MODEL_LEN"

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --dtype auto
