#!/usr/bin/env bash
# Stream-sample a DROID MiniWorld checkpoint.
# Default model is 1B. Set MODEL=0.5B or MODEL=3B to switch model scale.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL="${MODEL:-1B}" # choices: B, L, 0.5B, 1B, 3B
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the LeRobot DROID root}"
CKPT="${CKPT:?Set CKPT to a MiniWorld checkpoint}"
VAE_CKPT="${VAE_CKPT:?Set VAE_CKPT to Wan2.2_VAE.pth}"
SAMPLE_DIR="${SAMPLE_DIR:-${REPO_DIR}/samples/droid_${MODEL}}"
GPU="${GPU:-0}"
HEADCAST="${HEADCAST:-0}"
headcast_args=()
if [[ "${HEADCAST}" == "1" ]]; then
  headcast_args+=(--headcast)
fi
forcing_kv_args=()
if [[ "${KV_ACCELERATION:-off}" == "forcing-kv" ]]; then
  forcing_kv_args+=(--forcing_kv_threshold "${FORCING_KV_THRESHOLD:-0.8}"
    --forcing_kv_segments "${FORCING_KV_SEGMENTS:-6}"
    --forcing_kv_retention "${FORCING_KV_RETENTION:-0.33}")
  if [[ -n "${FORCING_KV_PROFILE:-}" ]]; then
    forcing_kv_args+=(--forcing_kv_profile "${FORCING_KV_PROFILE}")
  fi
  if [[ -n "${FORCING_KV_PROFILE_OUTPUT:-}" ]]; then
    forcing_kv_args+=(--forcing_kv_profile_output "${FORCING_KV_PROFILE_OUTPUT}")
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -m miniworld.sample \
  --dataset droid \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${CKPT}" \
  --vae_checkpoint "${VAE_CKPT}" \
  --sample_dir "${SAMPLE_DIR}" \
  --wm_model "${MODEL}" \
  --total_len "${TOTAL_LEN:-64}" \
  --df_chunk_size 2 \
  --df_ardiff_step "${DF_ARDIFF_STEP:-5}" \
  --stream_inflight_chunks "${STREAM_INFLIGHT_CHUNKS:-8}" \
  --stream_max_cache_chunks "${STREAM_MAX_CACHE_CHUNKS:-24}" \
  --stream_sink_size "${STREAM_SINK_SIZE:-1}" \
  --kv_acceleration "${KV_ACCELERATION:-off}" \
  --cfg_scale "${CFG_SCALE:-2.0}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS:-100}" \
  --sample_num_videos "${SAMPLE_NUM_VIDEOS:-50}" \
  "${headcast_args[@]}" "${forcing_kv_args[@]}"
