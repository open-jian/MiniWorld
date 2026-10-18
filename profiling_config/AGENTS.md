# MiniWorld experiment conventions (user instruction, 2026-10-17)

- Window size means sliding historical KV capacity only, measured in chunks. Exclude the fixed attention sink and the in-flight denoising region from the window. State both separately in configs, reports and plots.
- Preserve the original sink (1 latent) and in-flight region (8 chunks = 16 latent) unless the user explicitly authorizes a change. Do not silently change these to accommodate a window endpoint.
- For N chunks of sliding history, exact retained capacity is sink_frames + N * chunk_size. Window zero still keeps the sink and in-flight denoising region.
- Current authorized sweep: episode 22694 only; generated cached history (normal autoregression), compared with the completed GT cached-history sweep; N=0,4,...,120 chunks; same seeds, actions, denoising steps and evaluation frames. Two plot families: quality vs history-window chunks; per-frame quality with 31 lines per metric. Always identify the episode.
- Keep outputs/data out of the project code directory. User authorized all physical GPUs 0–7 for the current sweep on 2026-10-17; retain running work and distribute remaining windows across available GPUs.

- Current follow-up authorized: normal generated history, episodes 22694/46584/19464/27206/12835, seed bases 20261016/20261017/20261018 (effective seed = base + episode), historical windows 0–24 chunks inclusive, step 1; all 8 GPUs. Preserve all fixed sampler settings and evaluation length.
