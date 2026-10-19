"""Bounded, read-only head probes on a real MiniWorld streaming trajectory.

Uses saved real inputs (only the seed latent is read by the sampler), keeps
the production attention outputs, and stops after two selected states.
"""
from __future__ import annotations

import time
PROCESS_START = time.monotonic()

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class BudgetExpired(Exception):
    pass


class ProfileComplete(Exception):
    pass


def masks_for_queries(query_ids, n_past, start, end, grid_h, grid_w, chunk_tokens):
    """All masks index only the legal prefix for this query chunk."""
    tpf = grid_h * grid_w
    keys = torch.arange(n_past + end, device=query_ids.device)
    sink = keys < min(tpf, n_past)
    recent_start = max(min(tpf, n_past), n_past - 2 * chunk_tokens)
    regions = {
        "sink": sink,
        "older_history": (keys >= min(tpf, n_past)) & (keys < recent_start),
        "recent_history": (keys >= recent_start) & (keys < n_past),
        "earlier_inflight": (keys >= n_past) & (keys < n_past + start),
        "current_chunk": keys >= n_past + start,
    }
    inflight = keys >= n_past
    qxy, kxy = query_ids % tpf, keys % tpf
    local = ((qxy[:, None] // grid_w - kxy[None, :] // grid_w).abs() <= 2)
    local &= ((qxy[:, None] % grid_w - kxy[None, :] % grid_w).abs() <= 2)
    candidates = {
        "inflight_only": inflight[None, :],
        "sink_recent1": (inflight | sink | (keys >= max(0, n_past - chunk_tokens)))[None, :],
        "sink_recent2": (inflight | sink | (keys >= max(0, n_past - 2 * chunk_tokens)))[None, :],
        "spatial_r2": local | inflight[None, :],
        "current_chunk_only": (keys >= n_past + start)[None, :],
    }
    return regions, candidates


def output_metrics(reference, output):
    cosine = F.cosine_similarity(reference, output, dim=-1, eps=1e-12)
    mse = (output - reference).square().mean(dim=(-1, -2))
    energy = reference.square().mean(dim=(-1, -2))
    return {
        "cos_mean": cosine.mean(-1),
        "cos_p05": torch.quantile(cosine, 0.05, dim=-1),
        "mse": mse,
        "relative_rmse": (mse / energy.clamp_min(1e-20)).sqrt(),
    }


def self_test():
    torch.manual_seed(17)
    qids = torch.tensor([8, 11, 15])
    regions, candidates = masks_for_queries(qids, 24, 8, 16, 2, 2, 8)
    assert torch.stack(list(regions.values())).sum(0).eq(1).all()
    assert all(mask.shape[-1] == 40 for mask in candidates.values())
    assert candidates["inflight_only"][0, :24].sum() == 0
    assert candidates["inflight_only"][0, 24:].all()
    assert candidates["current_chunk_only"].sum() == 8
    assert candidates["sink_recent1"].sum() == 4 + 8 + 16
    assert candidates["sink_recent2"].sum() == 4 + 16 + 16
    q, k, v = torch.randn(2, 3, 8), torch.randn(2, 40, 8), torch.randn(2, 40, 8)
    scores = q @ k.transpose(-1, -2) / 8**0.5
    for mask in candidates.values():
        actual = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    # Overlapping sink/recent at startup still form a partition.
    regions, _ = masks_for_queries(torch.tensor([0, 3]), 4, 0, 8, 2, 2, 8)
    assert torch.stack(list(regions.values())).sum(0).eq(1).all()
    # Spatial test: near a corner, only four 2-D neighbours for radius two
    # on a 5x6 grid? Actual clipped support is 3x3 = 9, per historical frame.
    _, candidates = masks_for_queries(torch.tensor([0, 14]), 60, 0, 30, 5, 6, 60)
    assert candidates["spatial_r2"][0, :60].sum() == 18
    assert candidates["spatial_r2"][:, 60:].all()
    print("Self-test passed: legal prefixes, disjoint regions, mask equivalence, spatial boundaries.")


class HeadProfiler:
    def __init__(self, output_dir, metadata, deadline):
        self.output_dir = output_dir
        self.metadata = metadata
        self.deadline = deadline
        self.records = []
        self.context = None
        self.layer = -1
        self.done = set()
        self.full_cond_calls = 0
        self.cond_forwards = 0
        self.probe_seconds = 0.0
        self.jsonl = (output_dir / "records.jsonl").open("w")

    def check_budget(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired()

    def pre_hook(self, index):
        def hook(module, args):
            self.layer = index
        return hook

    def probe(self, q, k, v, chunk_tokens, production_output):
        if self.context is None:
            return
        self.check_budget()
        begin = time.monotonic()
        context = self.context
        n_past = k.shape[-2] - q.shape[-2]
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else v.dtype
        # Match the exact tensors supplied to production FlashAttention, then
        # accumulate reference statistics in FP32 to expose small differences.
        with torch.autocast("cuda", enabled=False):
            kk = k[0].to(dtype).float()
            vv = v[0].to(dtype).float()
            for label, chunk_index, timestep in context["chunks"]:
                self.check_budget()
                start = chunk_index * chunk_tokens
                end = min(start + chunk_tokens, q.shape[-2])
                grid_h, grid_w = self.metadata["latent_grid"]
                tpf = grid_h * grid_w
                ys = ((torch.arange(4, device=q.device) + .5) * grid_h / 4).long()
                xs = ((torch.arange(4, device=q.device) + .5) * grid_w / 4).long()
                spatial_ids = (ys[:, None] * grid_w + xs[None, :]).flatten()
                frame_starts = torch.arange(start, end, tpf, device=q.device)
                ids = (frame_starts[:, None] + spatial_ids[None, :]).flatten()
                ids = ids[ids < end]
                qq = q[0, :, ids].to(dtype).float()
                legal_end = n_past + end
                scores = (qq @ kk[:, :legal_end].transpose(-1, -2)) * q.shape[-1] ** -.5
                probs = scores.softmax(-1)
                reference = probs @ vv[:, :legal_end]
                regions, candidates = masks_for_queries(ids, n_past, start, end, grid_h, grid_w, chunk_tokens)
                mass = {name: probs[:, :, mask].sum(-1).mean(-1) for name, mask in regions.items()}
                metrics = {}
                for name, mask in candidates.items():
                    out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vv[:, :legal_end]
                    metrics[name] = output_metrics(reference, out)
                    metrics[name]["retained_key_fraction"] = mask.float().mean().expand(q.shape[1])
                    metrics[name]["retained_mass"] = (probs * mask).sum(-1).mean(-1)
                actual = production_output[0, :, ids].float()
                reference_error = output_metrics(reference, actual)["relative_rmse"]
                max_abs = (actual - reference).abs().amax(dim=(-1, -2))
                energy = reference.square().mean(dim=(-1, -2)).sqrt()
                # Transfer only tiny aggregate arrays, never attention maps.
                mass_cpu = {name: arr.cpu().tolist() for name, arr in mass.items()}
                metrics_cpu = {name: {key: arr.cpu().tolist() for key, arr in vals.items()} for name, vals in metrics.items()}
                reference_error, max_abs, energy = reference_error.cpu().tolist(), max_abs.cpu().tolist(), energy.cpu().tolist()
                for head in range(q.shape[1]):
                    record = {
                        "state": context["state"], "cache_frames": n_past // tpf,
                        "branch": context["branch"], "layer": self.layer, "head": head,
                        "chunk_index": chunk_index, "noise_label": label, "timestep": timestep,
                        "sampled_queries": len(ids), "allowed_keys": legal_end,
                        "mass": {name: vals[head] for name, vals in mass_cpu.items()},
                        "candidates": {name: {key: vals[head] for key, vals in stats.items()} for name, stats in metrics_cpu.items()},
                        "reference_check": {"relative_rmse": reference_error[head], "max_abs": max_abs[head]},
                        "reference_rms": energy[head],
                    }
                    self.records.append(record)
                    self.jsonl.write(json.dumps(record) + "\n")
            self.jsonl.flush()
        self.probe_seconds += time.monotonic() - begin

    def wrap_forward(self, original):
        def forward(x, t, y=None, **kwargs):
            self.check_budget()
            self.context = None
            is_commit = kwargs.get("return_kv", False)
            branch = "uncond" if kwargs.get("cond_drop") is not None else "cond"
            cache_frames = kwargs.get("current_position_offset", 0)
            if not is_commit and branch == "cond":
                self.cond_forwards += 1
                if cache_frames == 48:
                    self.full_cond_calls += 1
            state = None
            if not is_commit and x.shape[2] == 16:
                if cache_frames == 8:
                    state = "short"
                elif cache_frames == 48 and self.full_cond_calls >= 11:
                    state = "full"
            if state is not None and (state, branch) not in self.done:
                chunk_size = kwargs["chunk_size"]
                values = t[0].reshape(-1, chunk_size).mean(-1).float().cpu().tolist()
                order = sorted(range(len(values)), key=values.__getitem__)
                chosen = [("low", order[0]), ("mid", order[len(order)//2]), ("high", order[-1])]
                self.context = {"state": state, "branch": branch,
                                "chunks": [(label, i, values[i]) for label, i in chosen]}
                print(f"[Profile] {state}/{branch} cache={cache_frames}, chunks={self.context['chunks']}", flush=True)
            active = self.context
            try:
                result = original(x, t, y, **kwargs)
            finally:
                self.context = None
            if active is not None:
                self.done.add((state, branch))
                self.save("running")
                print(f"[Profile] collected {state}/{branch}; {len(self.records)} head records, {time.monotonic()-PROCESS_START:.1f}s total", flush=True)
                if len(self.done) == 4:
                    raise ProfileComplete()
            return result
        return forward

    def save(self, status):
        self.jsonl.flush()
        self.metadata.update(elapsed_seconds=time.monotonic() - PROCESS_START,
                             probe_seconds=self.probe_seconds,
                             conditional_denoising_forwards=self.cond_forwards,
                             full_cache_conditional_forwards=self.full_cond_calls,
                             completed_states=sorted([list(x) for x in self.done]))
        payload = {"metadata": self.metadata, "status": status, "records": self.records}
        tmp = self.output_dir / "profile.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.output_dir / "profile.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deadline = PROCESS_START + args.budget_seconds
    def expired(signum, frame):
        raise BudgetExpired()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
    manifest = json.loads((args.input_dir / "experiment_manifest.json").read_text())
    metadata = {
        "scope": "same-input per-head sensitivity on unchanged real generated-history trajectory; not rollout quality or speedup",
        "episode": manifest["episode"], "seed": manifest["seed"],
        "input_dir": str(args.input_dir), "checkpoint": manifest["args"]["checkpoint"],
        "source_video": manifest["source"], "actions": manifest["actions"],
        "budget_seconds": args.budget_seconds,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "production_source_sha256": hashlib.sha256((REPO / "miniworld/miniworld.py").read_bytes()).hexdigest(),
        "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "latent_grid": [15, 20], "layers": 28, "heads": 12,
        "weight_dtype": "float32", "attention_input_dtype": "bfloat16", "statistics_dtype": "float32",
        "query_sampling": "4x4 spatial stratum centers in both latent frames = 32 queries/chunk; same grid in every probe",
        "state_selection": "first 8-frame-cache forward; eleventh conditional forward after cache reaches 48 frames (two further commits)",
        "noise_selection": "lowest, middle-ranked and highest t chunks within each selected real forward; labels are relative, see actual t",
        "history_regions": "sink=first latent; recent=last four cached latents excluding sink; older=remaining committed history",
        "candidate_definitions": {
            "inflight_only": "keep all legal in-flight prefix, discard committed history",
            "sink_recent1": "sink + last 2 cached latent frames + all legal in-flight prefix",
            "sink_recent2": "sink + last 4 cached latent frames + all legal in-flight prefix",
            "spatial_r2": "centered radius-2 square in every cached frame, clipped to image boundary; keep all legal in-flight prefix",
            "current_chunk_only": "keep only the query's own two-frame chunk; diagnostic, also deletes earlier in-flight chunks",
        },
        "relative_rmse_definition": "sqrt(mean((candidate-full)^2)/mean(full^2)), over sampled queries and head dimensions",
        "timing_scope": "process startup + model/input load + baseline sampling to two probe states + probes; no VAE encode/decode or full video generation",
    }
    profiler = HeadProfiler(args.output_dir, metadata, deadline)
    status = "error"
    hooks = []
    original_attention = None
    original_forward = None
    try:
        torch.set_num_threads(4)
        torch.manual_seed(manifest["seed"])
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        import miniworld.miniworld as model_module
        raw = torch.load(manifest["args"]["checkpoint"], map_location="cpu", weights_only=True)
        weights = raw.get("ema_model", raw.get("model", raw))
        options = dict(manifest["args"])
        options["latent_size"] = (15, 20)
        options["latent_frames"] = weights["net.feat_rope.freqs_cos"].shape[0] // 300
        options["wm_use_checkpoint"] = False
        model = build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights, strict=True)
        del weights, raw
        model = model.cuda().eval()
        inputs = torch.load(args.input_dir / "prepared_inputs.pt", map_location="cpu", weights_only=True)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        torch.cuda.synchronize()
        metadata["load_seconds"] = time.monotonic() - PROCESS_START
        metadata["sampler_args"] = options
        print(f"[Profile] real inputs/model loaded in {metadata['load_seconds']:.1f}s", flush=True)
        original_attention = model_module._chunkwise_flash_attention
        def attention(q, k, v, chunk_tokens):
            result = original_attention(q, k, v, chunk_tokens)
            profiler.probe(q, k, v, chunk_tokens, result)
            return result
        model_module._chunkwise_flash_attention = attention
        hooks = [block.attn.register_forward_pre_hook(profiler.pre_hook(i)) for i, block in enumerate(model.net.blocks)]
        original_forward = model.net.forward_with_cache
        model.net.forward_with_cache = profiler.wrap_forward(original_forward)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.generate_eval_latents_streaming(
                inputs["latents"], inputs["cond_seq"], total_len=inputs["latents"].shape[2],
                history_len=1, max_cache_chunks=24, inflight_chunks=8, sink_frames=1,
                noise=inputs["noise"], stream_decoder=None, collect_stream_timing=False,
            )
        status = "trajectory_finished"
    except ProfileComplete:
        status = "complete"
    except BudgetExpired:
        status = "budget_exhausted"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if original_attention is not None:
            model_module._chunkwise_flash_attention = original_attention
        if original_forward is not None:
            model.net.forward_with_cache = original_forward
        for hook in hooks:
            hook.remove()
        profiler.save(status)
        profiler.jsonl.close()
        print(f"[Profile] {status}: {len(profiler.records)} head records in {time.monotonic()-PROCESS_START:.1f}s", flush=True)


if __name__ == "__main__":
    main()
