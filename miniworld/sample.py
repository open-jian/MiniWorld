"""MiniWorld streaming inference entry point."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader

from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.conditioning.trajectories import SUPPORTED_TRAJECTORIES, build_custom_trajectory, load_init_image
from miniworld.data.droid import LeRobotActionDataset
from miniworld.data.re10k import RealEstate10KDataset
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae, print0, vae_encode


SAMPLE_HISTORY_LEN = 1
WM_MODEL_CHOICES = ("B", "L", "0.5B", "1B", "3B")


def write_video(path: str, frames: torch.Tensor, fps: int) -> None:
    """Write ``(T,H,W,C)`` uint8 video through a local temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        torchvision.io.write_video(tmp_path, frames.cpu(), fps=fps)
        target.write_bytes(Path(tmp_path).read_bytes())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_dataset(args: argparse.Namespace):
    """Build an evaluation dataset with deterministic clips."""
    num_raw_frames = 4 * (int(args.total_len) - 1) + 1
    if args.dataset == "droid":
        return LeRobotActionDataset(
            root=args.data_root,
            num_frames=num_raw_frames,
            frame_interval=1,
            resize_hw=(args.resize_h, args.resize_w),
            camera_views=args.action_camera_views.split(","),
            action_keys=args.action_keys.split(","),
            action_norm="q01q99",
            randomize=False,
            color_aug=False,
            require_success=True,
            max_keep=args.sample_num_videos,
        )
    return RealEstate10KDataset(
        dataset_paths=[args.data_root],
        num_frames=num_raw_frames,
        frame_interval=1,
        resize_hw=(args.resize_h, args.resize_w),
        randomize=False,
        color_aug=False,
        filter_cache_dir=args.dataset_filter_cache_dir,
        max_keep=args.sample_num_videos,
        return_pose=True,
        pose_dir=args.pose_dir,
    )


def resolve_conditioning(args: argparse.Namespace, dataset) -> None:
    """Populate conditioning dimensions on args."""
    if args.dataset == "droid":
        args.use_action_cond = True
        args.use_pose_cond = False
        args.cond_dim = 4 * int(dataset.d_action)
        args.cond_per_token = False
    else:
        args.use_action_cond = False
        args.use_pose_cond = True
        args.cond_dim = 4 * 6 * 2 * int(args.pose_enc_freq)
        args.cond_per_token = True


def resolve_custom_re10k_conditioning(args: argparse.Namespace) -> None:
    """Configure pose conditioning for procedural RE10K trajectories."""
    args.use_action_cond = False
    args.use_pose_cond = True
    args.cond_dim = 4 * 6 * 2 * int(args.pose_enc_freq)
    args.cond_per_token = True


def build_denoiser(args: argparse.Namespace):
    """Create a denoiser configured for sampling."""
    cfg_kwargs = vars(args).copy()
    cfg_kwargs.update(
        latent_size=(args.resize_h // args.spatial_downsample, args.resize_w // args.spatial_downsample),
    )
    if getattr(args, 'kv_acceleration', 'off') == 'forcing-kv':
        from miniworld.forcing_kv import checkpoint_identity
        cfg_kwargs['forcing_kv_model_identity'] = checkpoint_identity(args.checkpoint)
    cfg = DenoiserConfig(**cfg_kwargs)
    return build_denoiser_from_mode(cfg)


def read_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Read wrapped training checkpoints or exported tensor state dicts."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        weights = ckpt.get("ema_model", ckpt.get("model"))
        if weights is not None:
            return weights, ckpt.get("meta", {})
        if ckpt and all(isinstance(value, torch.Tensor) for value in ckpt.values()):
            return ckpt, {}
    raise ValueError(f"Checkpoint must contain ema_model/model or be a tensor state dict: {path}")


def resolve_latent_frames(weights: Dict[str, torch.Tensor], meta: Dict[str, object], args: argparse.Namespace) -> int:
    """Recover the latent window the checkpoint was built with.

    This sizes the RoPE tables, so it is a property of the checkpoint rather
    than a sampling choice. The rollout length is ``--total_len``, and the
    active attention window follows the streaming chunk counts.
    """
    for key in ("latent_frames", "trained_num_frames"):
        frames = int(meta.get(key, 0)) if isinstance(meta, dict) else 0
        if frames > 0:
            return frames
    # Metadata-free checkpoint: read the extent off the RoPE table itself.
    freqs = weights.get("net.feat_rope.freqs_cos")
    tokens_per_frame = (args.resize_h // args.spatial_downsample) * (args.resize_w // args.spatial_downsample)
    if freqs is not None and tokens_per_frame > 0 and freqs.shape[0] % tokens_per_frame == 0:
        return int(freqs.shape[0] // tokens_per_frame)
    raise ValueError(
        "Cannot determine the checkpoint's latent frame count: it has no metadata "
        "and no RoPE table. Re-export it from a MiniWorld training run."
    )


def resolve_wm_model(meta: Dict[str, object], args: argparse.Namespace) -> str:
    """Prefer the model scale recorded in the checkpoint over ``--wm_model``.

    Checkpoints written before this metadata existed fall back to the CLI value,
    where a wrong scale surfaces as a state dict mismatch at load time.
    """
    recorded = str(meta.get("wm_model", "")) if isinstance(meta, dict) else ""
    if recorded not in WM_MODEL_CHOICES:
        return args.wm_model
    if recorded != args.wm_model:
        print0(f"[Checkpoint] wm_model={recorded} from metadata overrides --wm_model {args.wm_model}")
    return recorded


def resolve_teacache(args: argparse.Namespace) -> None:
    """Resolve launcher defaults after CLI overrides and checkpoint metadata."""
    other_acceleration = args.headcast or args.kv_acceleration != "off"
    if args.teacache is None:
        args.teacache = bool(
            args.teacache_auto and args.wm_model == "1B"
            and not other_acceleration and not args.teacache_calibration
        )
    if not (args.teacache or args.teacache_calibration):
        print0("[TeaCache] disabled")
        return
    if other_acceleration:
        raise ValueError("TeaCache must be used separately from HeadCast/KV acceleration")
    if args.teacache and not args.teacache_calibration and not args.teacache_profile:
        if args.wm_model != "1B":
            raise ValueError("TeaCache for models other than 1B requires --teacache_profile")
        args.teacache_profile = str(
            Path(__file__).resolve().parents[1] / "profiling_config" / "teacache_profiles"
            / f"miniworld_1b_{args.dataset}.json"
        )
    from miniworld.teacache import DEFAULT_TEACACHE_THRESHOLD
    profile = json.loads(Path(args.teacache_profile).read_text()) if args.teacache_profile else {}
    if args.teacache_threshold is None:
        args.teacache_threshold = profile.get("threshold", DEFAULT_TEACACHE_THRESHOLD)
    mode = "calibration (no reuse)" if args.teacache_calibration else "enabled"
    print0(f"[TeaCache] {mode}, threshold={args.teacache_threshold}, "
           f"profile={args.teacache_profile or '(none)'}")


def load_weights(weights: Dict[str, torch.Tensor], denoiser: torch.nn.Module) -> None:
    """Load EMA weights into the denoiser."""
    denoiser.load_state_dict(weights, strict=True)
    print0("[Checkpoint] loaded: all keys matched")


def warn_on_timestep_shift_mismatch(meta: Dict[str, object], denoiser: torch.nn.Module) -> None:
    """Warn when sampling uses a different timestep shift than training.

    Checkpoints predating this metadata are skipped.
    """
    trained = float(meta.get("timestep_shift", -1.0)) if isinstance(meta, dict) else -1.0
    if trained <= 0:
        return
    active = float(denoiser.timestep_shift)
    if abs(active - trained) < 1e-4:
        return
    print0(
        f"[Warning] timestep shift mismatch: checkpoint trained with {trained:.4f}, "
        f"sampling with {active:.4f}. Pass --timestep_shift {trained:.4f} to match, "
        "or check that resize_h/resize_w, spatial_downsample and df_chunk_size "
        "are the same as in training."
    )


def summarize_timing(meta: Dict[str, object]) -> Dict[str, float]:
    """Summarize chunk timing metadata produced by streaming inference."""
    timing = meta.get("stream_timing", {}) if isinstance(meta, dict) else {}
    dit_events = [e for e in timing.get("dit_chunk_events", []) if e.get("generated", False)]
    vae_events = [e for e in timing.get("vae_chunk_events", []) if e.get("generated", False)]
    dit_times = [float(e["complete_sec"]) for e in dit_events if "complete_sec" in e]
    total_times = [float(e["end_sec"]) for e in vae_events if "end_sec" in e]
    vae_durations = [float(e["duration_sec"]) for e in vae_events if "duration_sec" in e]

    def mean_interval(values):
        if len(values) <= 1:
            return float("nan")
        return float(np.diff(np.asarray(values, dtype=np.float64)).mean())

    steady_dit = mean_interval(dit_times)
    steady_total = mean_interval(total_times)
    steady_vae = float(np.asarray(vae_durations[1:] or vae_durations, dtype=np.float64).mean()) if vae_durations else float("nan")
    return {
        "first_chunk_latency_total": total_times[0] if total_times else float("nan"),
        "steady_total_chunks_per_sec": 1.0 / steady_total if steady_total > 0 else float("nan"),
        "steady_dit_chunks_per_sec": 1.0 / steady_dit if steady_dit > 0 else float("nan"),
        "steady_vae_chunks_per_sec": 1.0 / steady_vae if steady_vae > 0 else float("nan"),
    }


def iter_custom_re10k_batches(args: argparse.Namespace, device: torch.device) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield batches for image-to-video sampling with procedural camera poses."""
    if args.dataset != "re10k":
        raise ValueError("--custom_camera_trajectory is only supported with --dataset re10k")
    if args.init_image is None:
        raise ValueError("--custom_camera_trajectory requires --init_image")

    image = load_init_image(args.init_image, args.resize_h, args.resize_w)
    videos = image.unsqueeze(0).unsqueeze(0).to(device)  # (B=1, T=1, H, W, C)
    num_raw_pose_frames = 4 * (int(args.total_len) - 1) + 1
    poses = build_custom_trajectory(
        args.custom_camera_trajectory,
        num_frames=num_raw_pose_frames,
        focal_norm=args.trajectory_focal_norm,
        magnitude=args.trajectory_magnitude,
    ).unsqueeze(0).to(device)
    if args.init_pose is not None:
        init_pose = torch.load(args.init_pose, map_location="cpu", weights_only=False)
        if init_pose.dim() == 2:
            init_pose = init_pose[0]
        if init_pose.numel() == 18:
            init_pose = torch.cat([init_pose[:4], init_pose[6:]], dim=0)
        if init_pose.numel() != 16:
            raise ValueError(f"--init_pose must contain a 16D pose or RE10K 18D pose, got {tuple(init_pose.shape)}")
        init_k = init_pose[:4].to(device=device, dtype=poses.dtype)
        base_focal = max(float(args.trajectory_focal_norm), 1e-6)
        scale = poses[..., 0] / base_focal
        poses[..., 0] = init_k[0] * scale
        poses[..., 1] = init_k[1] * scale
        poses[..., 2] = init_k[2]
        poses[..., 3] = init_k[3]
    for _ in range(args.sample_num_videos):
        yield {"videos": videos, "poses": poses, "actions": None}


def parse_args() -> argparse.Namespace:
    """Parse public MiniWorld sampling arguments."""
    parser = argparse.ArgumentParser("Sample MiniWorld")
    parser.add_argument("--dataset", choices=["droid", "re10k"], required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--pose_dir", default=None)
    parser.add_argument("--dataset_filter_cache_dir", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae_checkpoint", required=True)
    parser.add_argument("--sample_dir", default="samples")
    parser.add_argument("--sample_num_videos", type=int, default=8)

    parser.add_argument(
        "--wm_model",
        choices=list(WM_MODEL_CHOICES),
        default="1B",
        help="Model scale. Ignored when the checkpoint records its own.",
    )
    parser.add_argument("--resize_h", type=int, default=240)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--action_camera_views", default="exterior_image_1_left")
    parser.add_argument("--action_keys", default="cartesian_position,gripper_position")
    parser.add_argument("--pose_enc_freq", type=int, default=15)
    parser.add_argument("--init_image", default=None)
    parser.add_argument("--init_pose", default=None)
    parser.add_argument("--custom_camera_trajectory", choices=SUPPORTED_TRAJECTORIES, default=None)
    parser.add_argument(
        "--trajectory_magnitude",
        type=float,
        default=3.0,
        help="Camera motion scale. The trajectory is spread over --total_len, so "
             "longer rollouts need a proportionally larger value for the same speed.",
    )
    parser.add_argument(
        "--trajectory_focal_norm",
        type=float,
        default=0.5,
        help="Normalized focal length; 0.5 matches typical RealEstate10K intrinsics.",
    )

    # --latent_frames is deliberately absent: it sizes the RoPE tables and is
    # therefore fixed by the checkpoint. Rollout length is --total_len.
    parser.add_argument("--total_len", type=int, default=64)
    parser.add_argument(
        "--history_len",
        type=int,
        default=1,
        help="Clean visual context frames at the start of the rollout; must be > 0.",
    )
    parser.add_argument("--latent_channels", type=int, default=48)
    parser.add_argument("--spatial_downsample", type=int, default=16)
    parser.add_argument("--wm_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--wm_use_qknorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wm_use_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wm_adaln_mode", dest="adaln_mode", choices=["adaln_lora", "fully_shared", "per_block"], default="adaln_lora")
    parser.add_argument("--wm_cond_dropout_prob", dest="cond_dropout_prob", type=float, default=0.1)
    parser.add_argument("--df_chunk_size", type=int, default=2)
    parser.add_argument("--df_ardiff_step", type=int, default=5)
    parser.add_argument(
        "--timestep_baseshift",
        type=float,
        default=2.667,
        help="SD3-style timestep shift at 600 tokens/chunk (240x320, chunk_size=2); "
             "scaled by sqrt(chunk_size*H_lat*W_lat/600). Must match training.",
    )
    parser.add_argument(
        "--timestep_shift",
        type=float,
        default=-1.0,
        help="Absolute timestep shift; overrides --timestep_baseshift when > 0.",
    )
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval_min", type=float, default=0.2)
    parser.add_argument("--cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--stream_inflight_chunks", type=int, default=8)
    parser.add_argument("--stream_max_cache_chunks", type=int, default=24)
    parser.add_argument("--stream_sink_size", type=int, default=1)
    parser.add_argument("--headcast", action="store_true",
                        help="Enable experimental one-time head classification and grouped history attention (default: off).")
    parser.add_argument("--teacache", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable independent TeaCache trunk-residual reuse (direct Python default: off).")
    parser.add_argument("--teacache_auto", action="store_true",
                        help="Launcher default: enable TeaCache for 1B unless another acceleration mode is selected. "
                             "Explicit --teacache/--no-teacache takes precedence.")
    parser.add_argument("--teacache_profile", default="",
                        help="MiniWorld calibration JSON; when enabled, 1B defaults to the selected dataset's bundled profile.")
    parser.add_argument("--teacache_threshold", type=float, default=None,
                        help="Override the profile refresh threshold; bundled profiles and the fallback use 0.4. "
                             "Lower it (e.g. 0.2, 0.1 or 0.05) if quality loss or visible drift is too large; "
                             "smaller values refresh more often and reduce speedup.")
    parser.add_argument("--teacache_max_reuse_steps", type=int, default=None,
                        help="Override the profile consecutive reuse cap; 0 disables the cap.")
    parser.add_argument("--teacache_calibration", action="store_true",
                        help="Collect input/output changes with complete computation; never reuse.")
    parser.add_argument("--teacache_trace_output", default="",
                        help="Enable per-forward decision/frame tracing and write JSON for TeaCache or calibration.")
    parser.add_argument("--teacache_trace", action="store_true",
                        help="Collect per-forward TeaCache decisions in last_eval_meta (default: off).")
    parser.add_argument("--headcast_tile_height", type=int, default=5)
    parser.add_argument("--headcast_tile_width", type=int, default=5)
    parser.add_argument("--headcast_recent_chunks", type=int, default=2)
    parser.add_argument("--headcast_max_relative_rmse", type=float, default=0.10)
    parser.add_argument("--headcast_min_cosine", type=float, default=0.95)
    parser.add_argument("--headcast_query_samples_per_chunk", type=int, default=32)
    parser.add_argument("--kv_acceleration", choices=("off", "dense", "headcast", "headcast-source", "dummy", "combined", "forcing-kv"),
                        default="off", help="Opt-in compact KV backend and sparse current/history attention.")
    parser.add_argument("--kv_calibration_frames", type=int, default=8)
    parser.add_argument("--kv_max_relative_rmse", type=float, default=0.10)
    parser.add_argument("--kv_min_cosine", type=float, default=0.95)
    parser.add_argument("--kv_spatial_score_threshold", type=float, default=0.755)
    parser.add_argument("--kv_query_samples_per_chunk", type=int, default=32)
    parser.add_argument("--kv_dummy_frames", type=int, default=1)
    parser.add_argument("--kv_recent_frames", type=int, default=4)
    parser.add_argument("--kv_tile_height", type=int, default=5)
    parser.add_argument("--kv_tile_width", type=int, default=5)
    parser.add_argument("--kv_dummy_fraction", type=float, default=0.5)
    parser.add_argument("--forcing_kv_threshold", type=float, default=0.8)
    parser.add_argument("--forcing_kv_segments", type=int, default=6)
    parser.add_argument("--forcing_kv_retention", type=float, default=0.33,
                        help="Fraction of candidate historical segments retained at each clean commit.")
    parser.add_argument("--forcing_kv_profile", default="",
                        help="Optional compatible MiniWorld head profile; skips online calibration.")
    parser.add_argument("--forcing_kv_profile_output", default="",
                        help="Export the head profile from online calibration for subsequent inference.")
    parser.add_argument("--save_fps", type=int, default=8)
    parser.add_argument("--benchmark_stream_timing", action="store_true")
    parser.add_argument("--benchmark_no_save", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    """Run streaming sampling."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    use_custom_re10k = args.custom_camera_trajectory is not None
    if use_custom_re10k:
        resolve_custom_re10k_conditioning(args)
        dataloader = iter_custom_re10k_batches(args, device)
    else:
        if args.data_root is None:
            raise ValueError("--data_root is required unless --custom_camera_trajectory is set")
        dataset = build_dataset(args)
        resolve_conditioning(args, dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    vae = load_wan22_vae(args)
    weights, meta = read_checkpoint(args.checkpoint)
    args.latent_frames = resolve_latent_frames(weights, meta, args)
    args.wm_model = resolve_wm_model(meta, args)
    print0(f"[Checkpoint] {args.checkpoint}: wm_model={args.wm_model}, latent_frames={args.latent_frames}")
    resolve_teacache(args)
    denoiser = build_denoiser(args).to(device).eval()
    load_weights(weights, denoiser)
    if int(meta.get("trained_num_frames", 0)) > 0:
        denoiser.trained_num_frames = int(meta["trained_num_frames"])
    warn_on_timestep_shift_mismatch(meta, denoiser)

    sample_root = Path(args.sample_dir)
    pred_root = sample_root / "pred"
    pred_root.mkdir(parents=True, exist_ok=True)
    timing_rows = []

    for idx, batch in enumerate(dataloader):
        if idx >= args.sample_num_videos:
            break
        videos = batch["videos"].to(device)
        poses = batch.get("poses")
        actions = batch.get("actions")
        if poses is not None:
            poses = poses.to(device)
        if actions is not None:
            actions = actions.to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            latents = vae_encode(vae, rearrange(videos, "b t h w c -> b c t h w").contiguous())
            _, c_latent, t_latent, h_lat, w_lat = latents.shape
            sample_history_len = args.history_len
            if use_custom_re10k:
                sample_history_len = SAMPLE_HISTORY_LEN
                full_latents = latents.new_zeros(latents.shape[0], c_latent, int(args.total_len), h_lat, w_lat)
                full_latents[:, :, :1] = latents[:, :, :1]
                latents = full_latents
                t_latent = int(args.total_len)
            cond_cfg = ConditioningConfig(args.use_pose_cond, args.use_action_cond, args.pose_enc_freq)
            cond_seq = build_cond_seq_for_batch(
                cfg=cond_cfg,
                poses=poses,
                actions=actions,
                t_latent=t_latent,
                h_lat=h_lat,
                w_lat=w_lat,
            )
            stream_decoder = StreamingVAEDecoder(vae)
            start = time.perf_counter()
            result = denoiser.generate_eval_latents_streaming(
                latents,
                cond_seq,
                total_len=args.total_len,
                history_len=sample_history_len,
                max_cache_chunks=args.stream_max_cache_chunks,
                inflight_chunks=args.stream_inflight_chunks,
                sink_frames=args.stream_sink_size,
                stream_decoder=stream_decoder,
                collect_stream_timing=args.benchmark_stream_timing,
            )
            gen_sec = time.perf_counter() - start
            pred_latents, pred_rgb = result

        if args.teacache or args.teacache_calibration:
            summary = denoiser.last_eval_meta['teacache']
            print0(
                f"[TeaCache] sample={idx}, mode={summary['mode']}, "
                f"reused={summary['reused_forwards']}/{summary['denoising_forwards']}, "
                f"full_kv_writes={summary['commit_forwards']}"
            )

        if args.headcast:
            summary = denoiser.last_eval_meta["headcast"]
            counts = summary["group_counts"]
            print0(
                f"[HeadCast] sample={idx}, status={summary['status']}, "
                f"temporal={counts['temporal']}, spatial={counts['spatial']}, "
                f"global={counts['global']}, "
                f"calibration_forwards={summary['calibration_forwards']}"
            )

        if args.kv_acceleration != "off":
            summary = denoiser.last_eval_meta["kv_acceleration"]
            print0(
                f"[KV acceleration] sample={idx}, mode={summary['mode']}, "
                f"status={summary['status']}, heads={summary['group_counts']}, "
                f"compact_calls={summary['compact_attention_calls']}, "
                f"cache_MiB={summary['final_cache_storage_bytes'] / 2**20:.1f}"
            )

        if args.benchmark_stream_timing:
            row = summarize_timing(denoiser.last_eval_meta)
            row.update({"sample_idx": idx, "gen_sec": gen_sec})
            timing_rows.append(row)

        if not args.benchmark_no_save:
            video = ((pred_rgb[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
            write_video(os.fspath(pred_root / f"sample_{idx:04d}.mp4"), video, fps=args.save_fps)
            print0(f"[Sample] wrote sample_{idx:04d}.mp4")

    if timing_rows:
        timing_path = sample_root / "throughput_timing.jsonl"
        with timing_path.open("w") as fp:
            for row in timing_rows:
                fp.write(json.dumps(row) + "\n")
        print0(f"[Benchmark] wrote {timing_path}")


if __name__ == "__main__":
    main()
