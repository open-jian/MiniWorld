#!/usr/bin/env bash
# DROID streaming throughput benchmark. Uses cfg=1.0 to measure raw throughput.
# Default model is 1B. Set MODEL=0.5B or MODEL=3B to switch model scale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL="${MODEL:-1B}" # choices: B, L, 0.5B, 1B, 3B
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the LeRobot DROID root}"
CKPT="${CKPT:?Set CKPT to a MiniWorld checkpoint}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
SAMPLE_DIR="${SAMPLE_DIR:-${REPO_DIR}/throughput/droid_${MODEL}}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m miniworld.sample \
  --dataset droid \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${CKPT}" \
  --vae_checkpoint "${VAE_CKPT}" \
  --sample_dir "${SAMPLE_DIR}" \
  --optimization_config "${OPTIMIZATION_CONFIG:-${REPO_DIR}/optimization_config.toml}" \
  --wm_model "${MODEL}" \
  --total_len "${TOTAL_LEN:-96}" \
  --df_chunk_size 2 \
  --df_ardiff_step "${DF_ARDIFF_STEP:-5}" \
  --stream_inflight_chunks "${STREAM_INFLIGHT_CHUNKS:-8}" \
  --stream_max_cache_chunks "${STREAM_MAX_CACHE_CHUNKS:-24}" \
  --stream_sink_size "${STREAM_SINK_SIZE:-1}" \
  --cfg_scale 1.0 \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-100}" \
  --sample_num_videos "${SAMPLE_NUM_VIDEOS:-20}" \
  --benchmark_stream_timing \
  --benchmark_no_save
