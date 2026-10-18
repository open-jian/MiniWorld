"""Prepare real DROID episodes for the paired framewise-conditioning benchmark.

Video decoding, action slicing/normalization, VAE encoding, and temporal action
packing use the repository implementations. Only selected MP4/parquet files and
the dataset's modality.json/stats.json metadata are needed, not a full dataset
index. Artifacts belong outside the source checkout.

Example::

    python profiling_config/prepare_framewise_conditioning.py \
        --episodes 002352 004354 007217 007616 \
        --data-root /data2/jian/data/DROID_long_horizon \
        --metadata-dir /data2/jian/benchmark/droid/meta \
        --vae-checkpoint /data2/jian/checkpoints/miniworld/Wan2.2_VAE.pth \
        --output-root /data2/jian/output/Miniworld/framewise_conditioning_four
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_framewise_conditioning import SAMPLING_SETTINGS, dump, git, hash_file  # noqa: E402
from miniworld.conditioning.actions import build_cond_seq_from_actions  # noqa: E402
from miniworld.data.droid import (  # noqa: E402
    LeRobotActionDataset, _gather_cols, build_action_normalizer, resolve_action_slices,
)
from miniworld.vae.codec import load_wan22_vae, vae_encode  # noqa: E402


class SelectedEpisodeDataset(LeRobotActionDataset):
    """Initialize only the state required by the official two decoding methods.

    This benchmark adapter intentionally avoids LeRobot's full-dataset index and
    filtering scan. It inherits _decode_clip and _load_actions unchanged, so their
    resizing, frame alignment, float64 normalization, and clipping stay identical.
    Keep these attributes in sync if either inherited method gains dependencies.
    """

    def __init__(self, metadata_dir, parquet_path, rgb_frames):
        self.num_frames = rgb_frames
        self.frame_interval = 1
        self.resize_hw = (240, 320)
        self.randomize = False
        self.color_aug = False
        self.parquet_path = parquet_path
        modality = json.loads((metadata_dir / "modality.json").read_text())
        stats = json.loads((metadata_dir / "stats.json").read_text())
        self.action_keys = ["cartesian_position", "gripper_position"]
        self.slices = resolve_action_slices(modality, self.action_keys)
        self.action_cols = _gather_cols(self.slices)
        self.norm_lo, self.norm_hi, self.norm_mode = build_action_normalizer(stats, self.slices, "q01q99")

    def _parquet_path(self, _episode):
        return self.parquet_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path,
                        default=REPO_ROOT / "profiling_config/droid_video_selection_50.json")
    parser.add_argument("--reference-manifest", type=Path,
                        help="Optional archived experiment manifest whose exact model/sampler arguments are reused")
    parser.add_argument("--latent-frames", type=int, default=125)
    parser.add_argument("--seed-base", type=int, default=20261016)
    args = parser.parse_args()
    if args.latent_frames < 17 or len(set(args.episodes)) != len(args.episodes):
        parser.error("Require at least 17 latent frames and unique episode IDs")
    if args.output_root.resolve().is_relative_to(REPO_ROOT.resolve()):
        parser.error("Write benchmark artifacts outside the source checkout")
    return args


def protocol_args(args):
    if args.reference_manifest:
        options = dict(json.loads(args.reference_manifest.read_text())["args"])
        for key, expected in SAMPLING_SETTINGS.items():
            if options.get(key) != expected:
                raise ValueError(f"Reference manifest changes {key}: {options.get(key)!r} != {expected!r}")
    else:
        # These are the archived 002352 settings; production defaults are not
        # consulted, so future default changes cannot silently alter this test.
        options = {
            **SAMPLING_SETTINGS,
            "dataset": "droid", "wm_model": "1B", "resize_h": 240, "resize_w": 320,
            "latent_channels": 48, "spatial_downsample": 16, "wm_mlp_ratio": 4.0,
            "wm_use_qknorm": True, "wm_use_checkpoint": False, "adaln_mode": "adaln_lora",
            "cond_dropout_prob": 0.1, "use_action_cond": True, "use_pose_cond": False,
            "cond_dim": 28, "cond_per_token": False, "pose_enc_freq": 15, "save_fps": 8,
            "action_camera_views": "exterior_image_1_left",
            "action_keys": "cartesian_position,gripper_position", "latent_frames": 64,
        }
    options["checkpoint"] = str(args.checkpoint.resolve())
    options["vae_checkpoint"] = str(args.vae_checkpoint.resolve())
    return options


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(4)
    options = protocol_args(args)
    selection = json.loads(args.selection_manifest.read_text())
    entries = {int(entry["episode_index"]): entry for entry in selection["videos"]}
    rgb_frames = 4 * (args.latent_frames - 1) + 1
    jobs = []
    for episode in args.episodes:
        entry = entries[episode]
        group = str(entry["target_frames"])
        source = args.data_root / group / "videos" / f"episode_{episode:06d}.mp4"
        actions = args.data_root / group / "actions" / f"episode_{episode:06d}.parquet"
        output_dir = args.output_root / f"episode_{episode:06d}" / "prepared"
        if (output_dir / "experiment_manifest.json").exists():
            raise FileExistsError(f"Prepared episode already exists: {output_dir}")
        if not source.is_file() or not actions.is_file():
            raise FileNotFoundError(f"Missing video or actions: {source}, {actions}")
        if rgb_frames > entry["actual_frames"]:
            raise ValueError(f"Episode {episode:06d} is shorter than the requested {rgb_frames} frames")
        dataset = SelectedEpisodeDataset(args.metadata_dir, actions, rgb_frames)
        jobs.append((episode, entry, source, actions, output_dir, dataset))

    vae = load_wan22_vae({"vae_checkpoint": args.vae_checkpoint})
    for episode, entry, source, actions_path, output_dir, dataset in jobs:
        started = time.perf_counter()
        video, frame_ids = dataset._decode_clip(source)
        if frame_ids != list(range(rgb_frames)):
            raise AssertionError("The benchmark must use consecutive source frames beginning at frame zero")
        actions = dataset._load_actions(episode, frame_ids).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gt_latents = vae_encode(vae, video.unsqueeze(0).cuda().permute(0, 4, 1, 2, 3).contiguous())
            cond_seq = build_cond_seq_from_actions(actions)
        if tuple(gt_latents.shape) != (1, 48, args.latent_frames, 15, 20):
            raise AssertionError(f"Unexpected VAE output shape: {tuple(gt_latents.shape)}")
        # The generator receives only the first observation. Separate full GT
        # latents are used solely for the optional shared-history forward checks.
        latents = gt_latents.clone()
        latents[:, :, 1:] = 0
        seed = args.seed_base + episode
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        noise = torch.randn_like(latents)
        data = {"latents": latents, "gt_latents": gt_latents, "cond_seq": cond_seq, "noise": noise}
        for name, tensor in data.items():
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"Non-finite prepared {name} for episode {episode:06d}")
        torch.cuda.synchronize()
        output_dir.mkdir(parents=True, exist_ok=True)
        pixels = ((video + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
        np.save(output_dir / "gt_rgb.npy", pixels)
        temporary = output_dir / "prepared_inputs.pt.tmp"
        torch.save({name: tensor.cpu() for name, tensor in data.items()}, temporary)
        temporary.replace(output_dir / "prepared_inputs.pt")
        manifest = {
            "episode": episode, "task": entry.get("task"), "seed": seed,
            "source": str(source.resolve()), "actions": str(actions_path.resolve()),
            "original_rgb_frames": entry["actual_frames"], "processed_rgb_frames": rgb_frames,
            "latent_frames": args.latent_frames, "args": options,
            "mode": "normal generated-history autoregression; GT initial latent only",
            "forward_gt_latents_separate": True,
            "action_normalization": "official q01q99, float64 before float32 conversion; 4 transitions per latent",
            "camera": entry["camera"], "resize_hw": [240, 320], "frame_interval": 1,
            "noise_dtype": str(noise.dtype), "vae_autocast_dtype": "torch.bfloat16",
            "noise_sha256": hashlib.sha256(noise.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "input_sha256": {str(path.resolve()): hash_file(path) for path in (
                source, actions_path, args.metadata_dir / "modality.json", args.metadata_dir / "stats.json",
            )},
            "reference_manifest": str(args.reference_manifest.resolve()) if args.reference_manifest else None,
            "reference_manifest_sha256": hash_file(args.reference_manifest) if args.reference_manifest else None,
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "prepare_seconds": time.perf_counter() - started,
        }
        dump(output_dir / "experiment_manifest.json", manifest)
        print(f"PREPARED episode={episode:06d} RGB={rgb_frames} latents={args.latent_frames}: {output_dir}", flush=True)
        del video, actions, data, latents, gt_latents, cond_seq, noise, pixels


if __name__ == "__main__":
    main()
