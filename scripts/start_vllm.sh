#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-14B-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-14b-awq}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve "$MODEL" \
--served-model-name "$SERVED_MODEL_NAME" \
--host 127.0.0.1 \
--port 8000 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-seqs "$MAX_NUM_SEQS" \
--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
--enable-prefix-caching \
--reasoning-parser qwen3 \
--structured-outputs-config.enable_in_reasoning=True