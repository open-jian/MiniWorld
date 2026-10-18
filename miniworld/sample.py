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
from miniworld.optimization_config import OptimizationConfig, load_optimization_config
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


def build_denoiser(args: argparse.Namespace, *, optimization_config: OptimizationConfig | None = None):
    """Create a denoiser configured for sampling."""
    cfg_kwargs = vars(args).copy()
    cfg_kwargs.update(
        latent_size=(args.resize_h // args.spatial_downsample, args.resize_w // args.spatial_downsample),
    )
    if optimization_config is not None:
        cfg_kwargs["optimization_config"] = optimization_config
    cfg = DenoiserConfig(**cfg_kwargs)
    return build_denoiser_from_mode(cfg)


def read_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Read EMA weights and metadata without touching the model."""
    ckpt = torch.load(path, map_location="cpu")
    weights = ckpt.get("ema_model", ckpt.get("model"))
    if weights is None:
        raise ValueError(f"Checkpoint has neither ema_model nor model: {path}")
    return weights, ckpt.get("meta", {})


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
        "--optimization_config",
        default=None,
        help="Inference optimization TOML file; defaults to optimization_config.toml in this checkout. "
             "Read once per model; sampling/window/precision settings are separate.",
    )

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
    parser.add_argument("--save_fps", type=int, default=8)
    parser.add_argument("--benchmark_stream_timing", action="store_true")
    parser.add_argument("--benchmark_no_save", action="store_true")
    args = parser.parse_args()
    return args


def main() -> None:
    """Run streaming sampling."""
    args = parse_args()
    # Validate before loading data/checkpoints/VAE. Pass this same immutable
    # snapshot into the model so a mid-startup file edit cannot change the run.
    optimizations = load_optimization_config(args.optimization_config)
    print0(f"[Optimizations] {json.dumps(optimizations.snapshot(), ensure_ascii=False)}")
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
    denoiser = build_denoiser(args, optimization_config=optimizations).to(device).eval()
    load_weights(weights, denoiser)
    if int(meta.get("trained_num_frames", 0)) > 0:
        denoiser.trained_num_frames = int(meta["trained_num_frames"])
    warn_on_timestep_shift_mismatch(meta, denoiser)

    sample_root = Path(args.sample_dir)
    pred_root = sample_root / "pred"
    pred_root.mkdir(parents=True, exist_ok=True)
    # Save actual loaded values, not a later reread of a mutable config file.
    # 'requested' deliberately does not claim FlashAttention ran on an
    # unsupported device/dtype; the original forward guards still apply.
    (sample_root / "optimization_config.json").write_text(
        json.dumps(denoiser.net.optimization_config.snapshot(), indent=2) + "\n",
        encoding="utf-8",
    )
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
