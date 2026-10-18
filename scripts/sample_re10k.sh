#!/usr/bin/env bash
# Stream-sample a RealEstate10K MiniWorld checkpoint.
# Default model is 1B. Set MODEL=0.5B or MODEL=3B to switch model scale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL="${MODEL:-1B}" # choices: B, L, 0.5B, 1B, 3B
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the RealEstate10K video root}"
POSE_DIR="${POSE_DIR:?Set POSE_DIR to the RealEstate10K pose tensor root}"
CKPT="${CKPT:?Set CKPT to a MiniWorld checkpoint}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
SAMPLE_DIR="${SAMPLE_DIR:-${REPO_DIR}/samples/re10k_${MODEL}}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m miniworld.sample \
  --dataset re10k \
  --data_root "${DATA_ROOT}" \
  --pose_dir "${POSE_DIR}" \
  --checkpoint "${CKPT}" \
  --vae_checkpoint "${VAE_CKPT}" \
  --sample_dir "${SAMPLE_DIR}" \
  --optimization_config "${OPTIMIZATION_CONFIG:-${REPO_DIR}/optimization_config.toml}" \
  --wm_model "${MODEL}" \
  --total_len "${TOTAL_LEN:-64}" \
  --df_chunk_size 2 \
  --df_ardiff_step "${DF_ARDIFF_STEP:-5}" \
  --stream_inflight_chunks "${STREAM_INFLIGHT_CHUNKS:-8}" \
  --stream_max_cache_chunks "${STREAM_MAX_CACHE_CHUNKS:-24}" \
  --stream_sink_size "${STREAM_SINK_SIZE:-1}" \
  --cfg_scale "${CFG_SCALE:-2.0}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-100}" \
  --sample_num_videos "${SAMPLE_NUM_VIDEOS:-50}"
