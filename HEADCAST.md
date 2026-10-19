# HeadCast for MiniWorld

This optional inference path adapts HeadCast's output-based head classification
to MiniWorld's asynchronous diffusion chunks. It is **disabled by default** and
does not change training or checkpoint parameters.

## Enable

Use the same data and checkpoint environment variables as the standard sampling
commands in the README, then enable either launcher:

```bash
HEADCAST=1 bash scripts/sample_droid.sh
HEADCAST=1 bash scripts/sample_re10k.sh
```

`HEADCAST=0` is the default. For direct Python sampling, add `--headcast`.

| CLI option | Default |
| --- | ---: |
| `--headcast_tile_height` | 5 |
| `--headcast_tile_width` | 5 |
| `--headcast_recent_chunks` | 2 |
| `--headcast_max_relative_rmse` | 0.10 |
| `--headcast_min_cosine` | 0.95 |
| `--headcast_query_samples_per_chunk` | 32 |

## Calibration and routing

Generation starts with full attention. Once the committed history cache fills,
each CFG branch is calibrated once on a denoising forward. MiniWorld probes the
lowest-, middle-, and highest-noise chunks available in that window, sampling
up to 32 queries per chunk. These labels are relative to the actual chunk
timesteps; different chunks are denoised at different noise levels.

For each head, the candidate output must have relative RMSE at most 10% and
mean cosine similarity at least 0.95 against the full-attention output in
**every observed chunk, batch item, and CFG branch**. Relative RMSE is
`sqrt(mean((candidate - full)^2) / mean(full^2))`. Calibration returns the
original full-attention outputs. A shared assignment becomes available only
after all layers and both CFG branches finish; without CFG, only the
conditional branch is required.

Heads then follow this priority:

| Path | Committed history visible to a query |
| --- | --- |
| Temporal | The configured sink frames plus the most recent two history chunks, with full spatial coverage. With the default chunk size of two, the recent tail contains four latent frames. |
| Spatial | The query's fixed, non-overlapping 5×5 token tile across all history frames, including the sink frames. Edge tiles use their actual valid size. |
| Global | The full existing history window. |

Temporal wins when both candidates pass; otherwise Spatial is selected if it
passes, with Global as the fallback. The spatial mask used for calibration is
the same fixed-tile mask used during generation.

**Every path keeps the complete legal in-flight prefix:** the query's current
chunk and all earlier in-flight chunks. Future chunks remain invisible, and
attention within the current chunk remains bidirectional. Clean context prefill
and the extra `t=0` forward that commits finished chunks always use full
attention.

## Scope and reference

A fresh runtime is created for every rollout, including each new sample in a
sampling loop. A rollout that never fills the cache stays on full attention.
The final classification and counts are available in
`denoiser.last_eval_meta["headcast"]`.

The implementation changes which history tokens attention reads. It retains
the dense backing KV cache, existing eviction and RoPE handling, and trained
active-window limit; it does **not** reduce persistent KV storage.

On supported CUDA devices, the backend batches query chunks with explicit
per-chunk key lengths. Global and temporal heads share one attention call;
spatial heads use valid tiles packed once per forward. The FlashAttention
KV-cache API is used for reads only: no new K/V is passed to it. CPU, FP32,
and unsupported CUDA configurations use PyTorch SDPA.

The reference implementation is [the official HeadCast repository at commit
`5fb72a5507cb13972a712c407354895e1b6f7926`](https://github.com/sjlgaga/HeadCast/tree/5fb72a5507cb13972a712c407354895e1b6f7926).
The thresholds and defaults above are MiniWorld-specific choices, rather than
the official paper's settings. The multiple noise checks, shared conservative
CFG assignment, preserved in-flight prefix, and dense clean commits also adapt
the method to MiniWorld's streaming sampler.

## Validation

Run the small correctness and isolation tests in the MiniWorld environment:

```bash
python -m unittest discover -s tests -v
```

The paired real-model check uses the local prepared DROID episode 002352 by
default; `--input-dir` can select another directory containing the same
`experiment_manifest.json` and `prepared_inputs.pt` format. It runs one
72-latent-frame baseline/on pair with identical noise and conditions, enforcing
a 300-second process budget including model loading:

```bash
CUDA_VISIBLE_DEVICES=0 python profiling_config/validate_headcast.py \
  --output-dir /tmp/miniworld_headcast_validation
```

Use a fresh output directory. Timing includes online calibration but excludes
VAE decoding. The check records per-layer assignments, output errors, cache
warm-up equality, and a same-input replay after disabling HeadCast. One
episode does not establish video quality or a stable throughput improvement.

### Initial 1B result (2026-10-19)

Episode 002352, seed 20263368, CFG 2.0, 72 latent frames, 48 cached frames,
16 in-flight frames, BF16, RTX PRO 6000 Blackwell Max-Q. The final enabled
run reused the saved baseline's inputs, noise and output from an earlier
process. Both timings include complete latent generation and exclude VAE.

| Measurement | Result |
| --- | ---: |
| Original chunk-Flash baseline | 70.085 s |
| HeadCast with packing and batched launches | 70.771 s |
| Assigned temporal / spatial / global heads | 46 / 35 / 255 |
| Relative latent RMSE, all 72 frames | 4.20% |
| Relative latent RMSE, final 24 frames | 7.03% |
| First 48 frames | Bitwise equal |
| Same-input replay after disabling | Bitwise equal |

This run **does not demonstrate an overall speedup** (enabled latency was
approximately 1.0% higher). Calibration starts only after the history fills;
the first 48 output frames use the original path. Both CFG branches calibrated
once and the grouped path executed 2,950 layer calls. The two calibration
forwards took 628 ms and 347 ms, including their normal dense forward work;
these are not measurements of classification overhead alone.

All 23 CPU/CUDA regression tests passed. Decoded-video quality and performance
across additional episodes remain unmeasured. Keep this path opt-in while
evaluating those properties; persistent KV compression is a separate change.
