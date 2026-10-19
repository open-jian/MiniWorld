# Forcing-KV for MiniWorld

Forcing-KV is an independent, inference-only `--kv_acceleration forcing-kv`
mode. The default remains `off`. HeadCast and Dummy strategies have separate
implementations and retain their existing settings. Runtime/cache state belongs
to one generation; checkpoint weights and training forwards are unchanged.

The official repository is checked out at `/data2/jian/project/Forcing-KV`,
commit [`50ac1f454348e7f0a2e54bde71e1652b20c09fe6`](https://github.com/zju-jiyicheng/Forcing-KV/tree/50ac1f454348e7f0a2e54bde71e1652b20c09fe6).
MiniWorld imports its own implementation and does not require the official
repository's environment, checkpoints, precomputed Wan head lists or assets.

## Enable

```bash
KV_ACCELERATION=forcing-kv \
DATA_ROOT=/path/to/droid \
CKPT=/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt \
VAE_CKPT=/data2/jian/checkpoints/miniworld/Wan2.2_VAE.pth \
bash scripts/sample_droid.sh
```

The RealEstate10K launcher accepts the same switches. Direct sampling uses
`python -m miniworld.sample --kv_acceleration forcing-kv` with the usual model
and data arguments. Set `KV_ACCELERATION=off` / `--kv_acceleration off` for the
original sampler. Legacy `--headcast` cannot be combined with an enabled KV
mode. Approximation quality is evaluated separately from cache correctness.

## What the port implements

1. **Static/dynamic head allocation.** The score is attention mass on the
   current query chunk and its immediately preceding latent frame, divided by
   legal attention mass excluding the sink. Scores at least 0.8 identify static
   heads. The first layer uses the short static path, matching the release's
   exclusion of that block from dynamic grouping.
2. **Static context.** Keep the sink, the frame immediately before each query
   chunk, and that chunk. For later in-flight chunks the predecessor is inside
   the current window. Sink/predecessor overlap is counted once, and future
   chunks are never visible. The initial sink can still reside in the in-flight
   window before its first clean commit.
3. **Dynamic history.** Keep the sink, the latest committed frame, and selected
   segments from the preceding short candidate window. On a two-frame clean
   commit, candidates are the previous boundary frame and the first newly
   committed frame; the second new frame supplies the next comparison boundary.
   The previous selected dynamic slots are replaced, not recursively retained.
4. **Similarity selection once per clean commit.** The second layer computes
   post-RoPE K cosine similarity at corresponding token positions, averages
   within each flattened segment, and keeps the lowest-similarity segments.
   Later layers gather their own K/V using the same selected indices. CFG
   branches have independent scores, indices and cache contents. For batch > 1,
   minimum similarity across the batch lets a change in any sample compete for
   retention. If the proxy layer has no dynamic heads, its all-head keys supply
   the proxy. The small extra proxy storage is reported.
5. **Physical compaction and fast attention.** Dense backing histories are
   released. Persistent packed history, batched FlashAttention and fused
   current-QKV packing use the common engine primitives. No history similarity
   scan runs during ordinary noisy denoising forwards. The logical position
   counter stays independent of packed token offsets, and eviction performs
   MiniWorld's original RoPE time shift on retained non-sink keys.

Default `segments=6` divides MiniWorld's 300 tokens per frame into six contiguous
50-token segments. These are flattened segments, not 5x5 spatial tiles.
`retention=0.33` is a **keep fraction**, not a drop fraction. For two candidate
frames it keeps `round(2*6*0.33)=4` segments: 200 tokens per dynamic head.
At steady state, static heads store 600 historical tokens per head and dynamic
heads store 800, versus the original 14,400. Actual memory also includes model
weights, current frames and packed scratch; logical token savings are not whole
process memory savings.

## Online calibration and reusable profiles

Without a profile, the first eight committed latent frames use the original
full-history context. One calibration forward per CFG branch then samples low,
middle and high noise chunks, averaging head scores across those observations.
The allocation remains fixed for the rest of that generation.

To build a reusable MiniWorld profile, run one representative video:

```bash
KV_ACCELERATION=forcing-kv SAMPLE_NUM_VIDEOS=1 \
FORCING_KV_PROFILE_OUTPUT=/path/to/miniworld_forcing_heads.json \
DATA_ROOT=/path/to/droid CKPT=/path/to/model.pt VAE_CKPT=/path/to/vae.pth \
bash scripts/sample_droid.sh
```

Reuse it with `FORCING_KV_PROFILE=/path/to/miniworld_forcing_heads.json`. The
direct flags are `--forcing_kv_profile_output` and `--forcing_kv_profile`.
Profile input and output are mutually exclusive. Exporting multiple samples to
the same path replaces that file with each sample's calibration; use one sample
for a single offline profile. The export belongs to the explicitly enabled
calibration run and its I/O is included if generation is timed.

A loaded profile bypasses calibration and routes from the first denoising
forward. Any aligned clean input-context prefill remains dense.
It validates checkpoint identity, conditioning, model/grid/head dimensions,
chunk/window settings, CFG branches and classifier settings. The standard
entry points bind identity to checkpoint absolute path, size and modification
time; moving/replacing that checkpoint requires regenerating the profile or
using a caller-supplied content identity. Python API callers set
`DenoiserConfig.forcing_kv_model_identity` explicitly before exporting/loading.
Published Wan profiles are rejected. Classification does not promise that a
profile transfers without quality loss to a different scene.

| Direct option | Launcher variable | Default |
| --- | --- | --- |
| `--forcing_kv_threshold` | `FORCING_KV_THRESHOLD` | 0.8 |
| `--forcing_kv_segments` | `FORCING_KV_SEGMENTS` | 6 |
| `--forcing_kv_retention` | `FORCING_KV_RETENTION` | 0.33 |
| `--forcing_kv_profile` | `FORCING_KV_PROFILE` | no profile |
| `--forcing_kv_profile_output` | `FORCING_KV_PROFILE_OUTPUT` | no export |

The shared `--kv_calibration_frames` and `--kv_query_samples_per_chunk` control
online calibration (defaults 8 and 32). Segment count must divide the spatial
token count. All these options are inactive when the selected mode is `off`.

## Explicit adaptations from the official implementation

- The release's common Wan path processes one three-frame chunk at a time.
  MiniWorld processes multiple two-frame chunks at different noise levels.
  Static anchors are therefore query-chunk specific; dynamic heads also retain
  the entire legal in-flight prefix. Only committed history is dynamically
  compacted. Neither branch can attend to a future chunk.
- The source uses supplied offline Wan head lists and short history even before
  its switch point. MiniWorld can calibrate its own heads from full context or
  load a compatible profile. A loaded profile and online calibration have
  different early trajectories; equal final head lists do not imply equal
  complete videos.
- Similarity uses the released second-block/post-RoPE proxy and low-similarity
  ranking, generalized to the actual commit length. Initial conversion uses the
  same short candidate horizon. This is the released short-cache mechanism,
  not a claim that the paper's entire long history keeps 70% of all tokens.
- We share only selected positions across layers. Mutable state is request/CFG
  owned, unlike the release's class-global indices. Existing MiniWorld sliding
  position rebasing replaces the release's absolute-time RoPE convention.
- The official Krea path lacks dynamic cache pruning; it is not the source for
  this port. TeaCache and quantization are separate and are not enabled here.

## Implementation and validation

- `miniworld/forcing_kv.py`: configuration, rollout lifecycle, counters and
  offline profile activation; shares the existing calibration schedule.
- `miniworld/forcing_kv_policy.py`: head scoring, allocation and profile I/O.
- `miniworld/forcing_kv_cache.py`: physical histories, shared selection,
  append/eviction and static/dynamic attention.
- `profiling_config/evaluate_forcing_kv.py`: paired real-model evaluation,
  including streaming VAE decode and all-frame RGB quality measurements.

```bash
python -m unittest discover -s tests -p 'test_forcing_kv*.py' -v
```

Runtime evidence is in `denoiser.last_eval_meta['kv_acceleration']`: head
partition, calibration/profile state, actual compact-attention and clean-commit
calls, per-branch selection updates, owned cache storage and logical positions.
Correctness tests cover independent attention/selection oracles, position
rebasing, partial chunks, profile compatibility, and restoration of the original
output and parameters after turning the switch off. End-to-end quality and
timing are reported from actual rollouts, not inferred from attention FLOPs or
the paper's DiT-only speedups.

## Measured DROID check (2026-10-19)

Paired runs used the same input frame, actions and noise, with 125 latent frames
decoded to 497 RGB frames. E2E includes input preparation, initial VAE encoding,
diffusion sampling and streaming VAE decoding. Each pair ran on one RTX PRO
6000 Blackwell Max-Q; mode order was reversed between episodes. Model loading,
warmup, metrics and output export are excluded. Online calibration is included.

Both methods are evaluated against the recorded ground-truth video (GT).
Quality metrics use RGB frames 1–496, excluding conditioning frame 0. PSNR
uses MSE pooled over those frames; SSIM and LPIPS are per-frame means.
Generated FPS is `496 / E2E seconds`, with VAE decoding included in E2E.

| Episode | Method | E2E seconds ↓ | Generated FPS ↑ | PSNR vs GT ↑ | SSIM vs GT ↑ | LPIPS vs GT ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 002352 | Baseline | 133.295 | 3.721 | 23.813 dB | 0.86590 | 0.04675 |
| 002352 | Forcing-KV | 101.197 | 4.901 | 23.658 dB | 0.85670 | 0.05177 |
| 022694 | Baseline | 135.388 | 3.664 | 19.371 dB | 0.80451 | 0.14034 |
| 022694 | Forcing-KV | 96.399 | 5.145 | 19.647 dB | 0.80405 | 0.13239 |

The mean paired speedup is 1.361x, with 26.44% less elapsed time. Peak allocated
memory fell from about 18.48 GiB to 11.58/11.66 GiB. Forcing-KV saves 24.08%
and 28.80% of E2E time on 002352 and 022694, respectively. On 002352 all three
GT quality metrics worsen slightly. On 022694 PSNR improves by 0.275 dB and
LPIPS decreases by 0.00796, while SSIM decreases by 0.00045. Thus the GT quality
change depends on the scene. These are two scenes, one seed per scene and one
timing per mode, not a general speed or quality guarantee. The PDF additionally
reports differences from baseline generation, clearly separated from GT quality.

All 69 focused and regression tests passed. Real-model checks verified actual
physical cache routing, independent CFG state, identical inputs, unchanged
source during timing, and bitwise restoration of the original fixed probe after
turning the mode off. Both full baseline outputs also match the earlier baseline
latents bitwise. A separate bounded real-model probe verified offline activation
with empty history and bitwise online/profile-loaded equivalence at the same
48-frame cached state; it is not an offline-mode full-rollout speed measurement.

English and Chinese figures/reports are saved under
`/data2/jian/output/Miniworld/kv-cache-alg/6th_forcing_kv/` (PDF only).
Raw measurements, per-frame metrics and reproducible commands are under
`/home/jian/Documents/Codex/2026-10-19/hello/work/forcing_kv_migration/`.
