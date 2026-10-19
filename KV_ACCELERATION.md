# Compact KV acceleration for MiniWorld

This opt-in inference path combines early head classification, physically
shortened temporal KV histories, spatially packed attention, fused current-QKV
packing, and batched chunk-attention launches. It adapts engineering from HeadCast and Dummy
Forcing to MiniWorld's simultaneous denoising of multiple chunks. It is a
MiniWorld implementation with the differences described below; its kernels
and quality settings are not identical to either official release.

The default is **off**. The original checkpoint, training forward, sampling
schedule, conditioning and CFG scale remain compatible. Runtime state belongs
to one generation and is discarded after that generation.

## Enable and select a mode

The DROID and RealEstate10K launchers accept `KV_ACCELERATION`:

```bash
KV_ACCELERATION=headcast-source \
DATA_ROOT=/path/to/droid \
CKPT=/path/to/MiniWorld_1b_droid.pt \
VAE_CKPT=/path/to/Wan2.2_VAE.pth \
bash scripts/sample_droid.sh
```

Set `KV_ACCELERATION=off` to restore the original inference path. For direct
sampling, pass `--kv_acceleration headcast-source` to `python -m miniworld.sample`.
The legacy `HEADCAST=1` / `--headcast` option selects the earlier history-only
implementation documented in [HEADCAST.md](HEADCAST.md); it cannot be combined
with an enabled `--kv_acceleration` mode.

| Mode | Behavior |
| --- | --- |
| `off` | Original sampler and attention; creates no compact-cache runtime. |
| `dense` | Same full-context visibility, with batched chunk launches and persistent packed history. No head classification. Useful for isolating backend costs. |
| `headcast` | Output-based classification with sink, dummy, spatial, then global priority. |
| `headcast-source` | The released HeadCast cosine and spatial-score gates, with sink, dummy, spatial, then global priority. RRMSE remains diagnostic and does not veto assignments. |
| `dummy` | Global attention-mass ranking with a fixed dummy-head budget, followed by sink/local assignment for the remaining heads. Output errors are recorded but do not veto assignments. |
| `combined` | Tests sink, dummy, spatial and local candidates; selects the passing candidate with the lowest sampled mean key count, otherwise global. |
| `forcing-kv` | Independent Forcing-KV static/dynamic policy, sink + transition anchors, and clean-commit segment selection. See [FORCING_KV.md](FORCING_KV.md) for its separate defaults and profile workflow. |

`dummy` is the aggressive method ablation: its forced budget is not a
quality-validated operating point for MiniWorld. The output-based modes apply
the local attention-output thresholds described below; those thresholds do
not establish complete-video quality by themselves.

## Visibility and cache contents

The default 1B setup has 28 layers, 12 heads, 300 tokens per latent frame,
two frames per chunk, eight in-flight chunks and 48 committed history frames.
All frames in one query chunk are mutually visible. Later chunks are never
visible to earlier queries.

For a query chunk, the head groups read:

| Group | Keys visible to this query chunk |
| --- | --- |
| Global | All committed history and all current chunks up to and including this chunk. |
| Sink | The first resident sink frame(s) and this query chunk only. |
| Dummy | The immediately preceding configured number of frames and this query chunk. |
| Local | The immediately preceding configured local-history window and this query chunk. |
| Spatial | The same non-overlapping spatial tile in every frame of the legal history/current prefix. |

“Preceding” is relative to the query chunk, so it can include earlier
in-flight frames. A later chunk does not always read the last committed frame
as its dummy context. Spatial attention restricts both committed history and
the current prefix. A 5x5 tile means a fixed block of up to five latent rows
and five latent columns; it is not a sliding neighborhood centered on every
query. Edge tiles contain only valid positions.

Sink, dummy and local groups retain only the historical frames they need.
Global and spatial groups retain history across the configured temporal
window. Spatial history is rearranged into tiles for attention, without
deleting temporal frames.

## Calibration and defaults

Classification begins at the first denoising forward with at least eight
committed frames. It observes chunks at the lowest, middle and highest noise
levels of the current window, in both CFG branches when CFG is enabled.
Query positions are deterministic spatial samples, so calibration does not
consume the generation RNG. The head assignment is fixed after all layers
and branches finish this observation.

The output-based modes evaluate the exact inference masks against dense
attention in FP32, using the same low-precision Q/K/V inputs as the production
attention kernel. In `headcast` and `combined`, every selected candidate must
satisfy both thresholds for every observed chunk and batch item in both CFG
branches:

- Relative RMSE is at most `kv_max_relative_rmse`.
- Mean query-output cosine is at least `kv_min_cosine`.

`headcast-source` uses the released HeadCast gates instead:

- Sink and dummy require mean query-output cosine at least `kv_min_cosine`.
- Spatial requires mean cosine strictly greater than `kv_min_cosine`, and
  fifth-percentile query cosine minus five times absolute mean squared error
  at least `kv_spatial_score_threshold`.
- Candidates are tried in sink, dummy, spatial, then global order. Relative
  RMSE is reported but is not used for selection in this mode.

The source mode checks these gates for every observed chunk, batch item and
CFG branch. For the spatial score, percentile and MSE are paired within each
observation before taking the worst score. This multi-observation aggregation
is a MiniWorld adaptation; the official source evaluates one selected step
and batch item.

The classifier also reports the difference between the actual dense kernel
output and its independent FP32 reference. It keeps the dense calibration
output in the generation trajectory.

| Direct sampling option | Default | Meaning |
| --- | --- | --- |
| `--kv_acceleration` | `off` | Backend and policy mode. |
| `--kv_calibration_frames` | `8` | Minimum committed history before classification. |
| `--kv_max_relative_rmse` | `0.10` | Maximum sampled attention-output relative RMSE. |
| `--kv_min_cosine` | `0.95` | Minimum mean query-output cosine. |
| `--kv_spatial_score_threshold` | `0.755` | Spatial `p05 - 5*MSE` threshold in `headcast-source` mode. |
| `--kv_query_samples_per_chunk` | `32` | Deterministic query sampling budget per selected chunk. |
| `--kv_dummy_frames` | `1` | Dummy group's preceding-frame budget. |
| `--kv_recent_frames` | `4` | Local group's preceding-frame budget. |
| `--kv_tile_height`, `--kv_tile_width` | `5`, `5` | Non-overlapping spatial tile dimensions. |
| `--kv_dummy_fraction` | `0.5` | Forced dummy fraction in `dummy` mode only. |
| `--stream_sink_size` | `1` | Existing sampler sink size, also used by the sink group. |

If a short rollout never reaches calibration, it uses dense attention through
the new backend. The summary records this as warmup, rather than claiming
sparse routing occurred.

## Cache lifecycle and positions

After classification, each layer and CFG branch receives its own compact
cache. Dense backing histories are released. Clean `t=0` commits also use
the selected attention policy, append their new K/V, then update retained
history and packed buffers. Conditional and unconditional cache contents are
never shared.

Physical storage length does not define the RoPE position. The cache retains
the sampler's logical frame count. New query/key RoPE uses that logical
offset. On overflow, the sink stays at its original position; surviving
non-sink keys receive the same temporal RoPE shift as the original sampler.
This preserves MiniWorld's bounded position scheme when different head
groups store different numbers of frames.

Canonical keys retain their incoming precision for repeated RoPE shifts.
FlashAttention buffers hold the low-precision packed history and reusable
current-token scratch space. Historical gathering and casting happen after
cache updates, rather than on every denoising step. Append and eviction use
frame-range slices; per-group version counters keep an unchanged sink's
packed history valid when other groups change. BF16 canonical values alias
their packed storage when the layout permits, with updates materializing new
survivors before packed history is refreshed.

A stride-aware Triton kernel gathers current Q/K/V into group order and casts
them to the attention dtype in one pass. Its workspace belongs to the
generation, with separate buffers for each device, dtype and CUDA stream.
Sequential layer calls and CFG branches reuse that temporary workspace; their
historical cache contents remain separate. At batch 1, 12 heads, 4,800 current
tokens and head dimension 128, the BF16 workspace occupies about 44.2 MB
(42.2 MiB) for the entire request on one stream, rather than that amount per
layer. CPU execution and installations without Triton use the PyTorch gather
fallback.

Reported storage deduplicates aliased buffers and counts the shared current
workspace once across layers and CFG branches. It includes canonical
histories and persistent packed buffers; fewer retained history tokens do
not by themselves establish lower total memory.

The implementation is divided into:

- [kv_policy.py](miniworld/kv_policy.py): observations, fidelity metrics and
  head assignment.
- [kv_cache.py](miniworld/kv_cache.py): physical histories, logical positions,
  append/eviction and grouped attention.
- [kv_kernels.py](miniworld/kv_kernels.py): fused current-QKV head gather and
  dtype conversion, with a generation-owned workspace shared across layers.
- [kv_acceleration.py](miniworld/kv_acceleration.py): per-generation lifecycle,
  calibration, dense warmup and runtime reporting.
- [denoiser.py](miniworld/denoiser.py) and [miniworld.py](miniworld/miniworld.py):
  explicit inference-context plumbing.

## Official sources and MiniWorld adaptations

The audited sources are:

- [HeadCast, commit `5fb72a5507cb13972a712c407354895e1b6f7926`](https://github.com/sjlgaga/HeadCast/tree/5fb72a5507cb13972a712c407354895e1b6f7926).
- [Dummy Forcing, commit `28128ed6204dcb4c572bf8719955124a5f2dbe20`](https://github.com/csguoh/DummyForcing/tree/28128ed6204dcb4c572bf8719955124a5f2dbe20).

HeadCast's released classifier compares masked and full attention outputs,
then applies sink/dummy/spatial/global priority. The bundled inference
configurations use mean cosine thresholds of 0.95. Spatial classification
also requires fifth-percentile cosine minus five times absolute MSE to reach
0.755. `headcast-source` implements these gates on MiniWorld's exact routing
masks, while `headcast` and `combined` use relative-RMSE and mean-cosine bounds
instead. All three modes aggregate across selected noise levels and CFG
branches. The release computes a query-centered spatial diagnostic
as well, but its actual classifier uses the windowed inference output and its
executed spatial kernel returns only Grid A. MiniWorld uses one fixed grid
with valid-token buckets at image edges; it does not copy the release's
zero-padding of keys into the softmax. These choices change calibration
coverage and edge semantics while preserving the released thresholds in
`headcast-source`. See the [official classifier](https://github.com/sjlgaga/HeadCast/blob/5fb72a5507cb13972a712c407354895e1b6f7926/wan/modules/headcast.py#L175-L245)
and [executed spatial path](https://github.com/sjlgaga/HeadCast/blob/5fb72a5507cb13972a712c407354895e1b6f7926/wan/modules/spatial_attn.py#L190-L219).

Dummy Forcing's released implementation separately normalizes sink and
neighbor attention masses over all layers and heads, ranks the maximum of
the two, assigns a fixed number of dummy heads, and chooses sink or neighbor
for the remainder. MiniWorld's `dummy` mode reproduces that ranking rule,
with deterministic observations averaged over its selected chunks, all batch
items and both CFG branches. Earlier current chunks belong to the query's
logical history. The original single-batch score-region assumptions are not
copied. Its paper equations use unnormalized masses, while this port follows
the [released normalized ranking](https://github.com/csguoh/DummyForcing/blob/28128ed6204dcb4c572bf8719955124a5f2dbe20/wan/modules/dummyforcing.py#L28-L80).

The source's packed-attention mechanism gives dummy heads one preceding
frame, then combines sink and dummy heads into one attention call because
their histories have equal length. It uses Triton head extraction. This
MiniWorld backend currently handles sink and dummy separately: simultaneous
query chunks need different recent-frame offsets while sharing a fixed sink.
It batches query chunks with FlashAttention's cache interfaces and fuses
current-Q/K/V head gathering and casting in one Triton pass. Sink and dummy
attention still use separate launches; combining those launches is a
remaining optimization. Persistent group histories, shared packing scratch
and slice-based cache updates are MiniWorld-specific engineering choices;
the implementation does not claim kernel identity with the official source.
See [the source's attention path](https://github.com/csguoh/DummyForcing/blob/28128ed6204dcb4c572bf8719955124a5f2dbe20/wan/modules/causal_model.py#L129-L172)
and [Triton head extraction](https://github.com/csguoh/DummyForcing/blob/28128ed6204dcb4c572bf8719955124a5f2dbe20/wan/modules/extract_head_triton.py#L7-L100).

No mode implements TeaCache, transformer-block skipping, cross-layer KV
sharing, or reuse of stale current K/V across denoising steps. Those are
separate methods. The official paper's speedups are not measurements of this
MiniWorld implementation.

## Validation and reporting

Run correctness tests in the MiniWorld environment:

```bash
python -m unittest discover -s tests -p 'test_kv_*.py' -v
```

The real-model validation harness uses fixed prepared inputs and noise. A
baseline/combined pair, including causal VAE decoding for a tail comparison,
can be run with:

```bash
CUDA_VISIBLE_DEVICES=0 python profiling_config/validate_kv_acceleration.py \
  --output-dir /tmp/miniworld_kv_validation \
  --modes baseline,full
```

Here `full` names the combined mode. In this harness, `new-headcast` selects
the guarded new HeadCast mode; `headcast-source` selects its official-gate
variant; `headcast` alone selects the legacy implementation.
Use `--modes baseline,dense,new-headcast,headcast-source,dummy,full` for separate ablations,
and size the process budget for the number of selected runs. The default
process budget is 360 seconds. `--pairs 2` reverses execution order on the
second pair; `--decode-tail-latents 0` disables VAE export for a timing-only
run.

Per-generation details are available in
`denoiser.last_eval_meta["kv_acceleration"]`: actual group counts, calibration
observations, routed and clean-commit call counts, retained storage, logical
history length, and per-head fidelity diagnostics. The harness measures the
complete sampler and saturated-cache calls separately, records memory and
per-frame output differences, and checks that disabling acceleration after
an enabled run restores the original behavior. Whole-rollout speed, local
kernel speed and estimated savings should be reported separately.

## Measured results — 2026-10-19

One DROID episode, MiniWorld 1B, RTX PRO 6000 Blackwell Max-Q, BF16
autocast, 72 latent frames, 48-frame history, eight concurrent two-frame
chunks, and both CFG branches. Every candidate uses the same input and
noise as its same-process baseline. The complete latent sampler is timed,
including calibration, commits, eviction, and first-use kernel costs.
Loading, VAE decoding, and file export are excluded. Each validation job
completed in 216–238 seconds including these excluded activities.

| Mode | Paired runs / noise seeds | Whole sampler speedup | Sampler time saved | Saturated-cache interval speedup | Full latent RRMSE vs original |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | 1 / 1 | 1.011x | 1.10% | 1.037x | 0.92% |
| `headcast-source` | 2 / 2 | 1.071x | 6.59% | 1.194x | 13.18–14.70% |
| `dummy` | 2 / 2 | 1.297x | 22.88% | 1.639x | 40.97–42.18% |
| `combined` | 1 / 1 | 1.023x | 2.28% | 1.095x | 15.27% |

Multi-run timing entries are medians of paired ratios. RRMSE is a
trajectory difference, not a perceptual quality score. All-global dense
attention keeps identical visibility but changes low-precision reduction
order; it is not a bitwise-identical generated trajectory. Turning the
feature off after every enabled mode restored the original fixed-probe
output bitwise. All source hashes stayed unchanged during measurements.

`headcast-source` selected 130–131 reduced-context heads out of 336.
Its decoded tail preserved the overall scene but showed visible arm/gripper
position differences; tail pixel PSNR was 24.55–26.59 dB. Dummy's 168 forced
dummy heads plus sink/local heads produced larger robot and background
changes, with tail PSNR 17.13–19.05 dB. The latter is not an accepted quality
setting. No mode has been validated across multiple scenes or long horizons.

Peak allocated memory was 14.95 GiB for the original, 14.04–14.15 GiB for
`headcast-source`, 6.86–6.87 GiB for `dummy`, 16.69 GiB for `combined`, and
17.77 GiB for `dense`. Packed layouts can increase total allocation despite
removing canonical history, so the modes do not all reduce memory.

The implementation passes 50 CPU/CUDA tests covering independent attention
mask oracles, mixed heads, partial chunks and tiles, real RoPE eviction,
shared scratch storage, checkpoint/training compatibility and off isolation.
An additional independent read-only integration review found no blocking
issues. These checks establish implementation correctness, not acceptable
quality of every approximation.

The default remains `off`. `headcast-source` is the more promising tested
approximate policy; `dummy` exposes the speed/memory versus output-drift
tradeoff. This port does not demonstrate a quality-preserving 2x MiniWorld
speedup, nor does this limited study establish an absolute speed ceiling.
