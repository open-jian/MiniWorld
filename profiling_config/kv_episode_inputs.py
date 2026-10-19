"""Reproducible DROID inputs for paired KV-acceleration evaluation.

Timed inference uses ``load_episode_cpu(..., include_gt=False)`` followed by
``prepare_inputs``: read initial RGB and actions, resize, encode the initial
frame, and construct conditions/noise. Future GT decoding and all hashing are
evaluation work outside that clock. Call ``episode_manifest`` once before
timing to populate its immutable source-hash cache.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miniworld.conditioning.actions import build_cond_seq_from_actions
from miniworld.data.droid import (_apply_norm, _gather_cols, build_action_normalizer,
                                  resolve_action_slices)
from miniworld.vae.codec import vae_encode

DATA_ROOT = Path('/data2/jian/data/DROID_long_horizon')
META_ROOT = Path('/data2/jian/benchmark/droid/meta')
LEGACY_ROOT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')
DEFAULT_EPISODES = (2352, 3905, 18216, 9695, 22694)
ACTION_KEYS = ('cartesian_position', 'gripper_position')
RESIZE_HW = (240, 320)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def episode_manifest(episode, total_latents=125):
    """Source provenance and hashes; populate this outside the inference timer."""
    episode, total_latents = int(episode), int(total_latents)
    if total_latents < 1:
        raise ValueError('total_latents must be positive')
    selection_path = DATA_ROOT/'selection.json'
    selection = json.loads(selection_path.read_text())
    matches = [row for row in selection['videos'] if int(row['episode_index']) == episode]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one selected source for episode {episode}')
    row = matches[0]
    rgb_count = 4*(total_latents-1)+1
    if int(row['actual_frames']) < rgb_count:
        raise ValueError(f'Episode {episode} is too short for {rgb_count} RGB frames')
    video, actions = Path(row['source']), Path(row['parquet'])
    modality, stats = META_ROOT/'modality.json', META_ROOT/'stats.json'
    for path in (video, actions, modality, stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        'episode': episode, 'seed': 20261016+episode,
        'source_video': str(video), 'source_actions': str(actions),
        'source_video_sha256': sha256_file(video), 'source_actions_sha256': sha256_file(actions),
        'normalization_stats': str(stats), 'normalization_stats_sha256': sha256_file(stats),
        'modality_file': str(modality), 'modality_sha256': sha256_file(modality),
        'camera': row['camera'], 'task': row.get('task'),
        'source_rgb_frames': int(row['actual_frames']),
        'total_latents': total_latents, 'requested_rgb_frames': rgb_count,
        'reference_frame_indices': list(range(rgb_count)),
        'action_row_indices': list(range(rgb_count-1)),
        'resize_hw': list(RESIZE_HW), 'resize_implementation': 'decord.VideoReader(width=320,height=240)',
        'rgb_normalization': 'CPU FP32 uint8 / 255 -> *2 - 1, then device transfer; no augmentation',
        'action_keys': list(ACTION_KEYS), 'action_columns': [0, 1, 2, 3, 4, 5, 12],
        'action_normalization': 'training q01q99, float64 arithmetic then FP32; clipped [-1,1]',
        'conditioning': 'latent0=zeros(28); subsequent latent packs four consecutive normalized seven-dimensional actions',
        'initial_latent_policy': 'only initial RGB encoded; all future latent placeholders zero; no future GT generation input',
        'noise_policy': 'isolated CUDA torch.Generator(seed=20261016+episode), FP32 standard normal',
        'end_to_end_includes': ['initial RGB and action file reads', 'initial frame resize/normalization',
                                'device transfers', 'initial-frame VAE encoding', 'action conditioning',
                                'noise construction', 'generation with streaming RGB decoding'],
        'end_to_end_excludes': ['model loading', 'warmup', 'future GT decode', 'hashing',
                                'quality metrics', 'artifact encoding/writing'],
    }


def load_episode_cpu(episode, total_latents=125, *, include_gt=True, decode_threads=2):
    """Fresh source reads; full GT outside timing, only first frame inside timing.

    ``gt_rgb`` is uint8 THWC and contains either the complete requested reference
    or only its initial frame. The manifest always retains the full target length.
    ``actions`` is normalized FP32 [1, 4*(Tlatent-1), 7].
    """
    import pandas as pd
    from decord import VideoReader, cpu

    metadata = dict(episode_manifest(int(episode), int(total_latents)))
    reader = VideoReader(metadata['source_video'], ctx=cpu(0), width=RESIZE_HW[1],
                         height=RESIZE_HW[0], num_threads=int(decode_threads))
    count = metadata['requested_rgb_frames']
    if len(reader) < count:
        raise ValueError(f'Source decoder reports only {len(reader)} frames, need {count}')
    frame_ids = list(range(count)) if include_gt else [0]
    rgb = reader.get_batch(frame_ids).asnumpy()
    modality = json.loads(Path(metadata['modality_file']).read_text())
    stats = json.loads(Path(metadata['normalization_stats']).read_text())
    slices = resolve_action_slices(modality, ACTION_KEYS)
    columns = _gather_cols(slices)
    lo, span, mode = build_action_normalizer(stats, slices, mode='q01q99')
    dataframe = pd.read_parquet(metadata['source_actions'], columns=['action'])
    raw = np.stack(dataframe['action'].values).astype(np.float64)
    if raw.shape[0] < count-1:
        raise ValueError(f'Only {raw.shape[0]} action rows, need {count-1}')
    normalized = _apply_norm(raw[:count-1, columns], lo, span, mode)
    actions = torch.from_numpy(normalized).float().unsqueeze(0)
    metadata.update(decoded_rgb_frames=len(frame_ids), decoded_source_frames=len(reader),
                    source_fps=float(reader.get_avg_fps()), include_future_gt=bool(include_gt))
    return {'gt_rgb': rgb, 'actions': actions, 'manifest': metadata}


@torch.inference_mode()
def prepare_inputs(data, vae, device, seed=None):
    """Initial RGB/actions -> GPU tensors; no hashes or CPU/GPU synchronization."""
    device = torch.device(device)
    metadata = data['manifest']
    total_latents = int(metadata['total_latents'])
    seed = int(metadata['seed'] if seed is None else seed)
    initial = torch.from_numpy(np.ascontiguousarray(data['gt_rgb'][:1]))
    # Match training/archived inputs exactly: CUDA division rounds differently.
    video = initial.float().div(255).mul(2).sub(1)
    video = video.permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        first = vae_encode(vae, video)
    if first.ndim != 5 or first.shape[2] != 1:
        raise RuntimeError(f'Initial-frame VAE output has unexpected shape {tuple(first.shape)}')
    latents = first.new_zeros((first.shape[0], first.shape[1], total_latents, *first.shape[-2:]))
    latents[:, :, :1].copy_(first)
    actions = data['actions'].to(device=device)
    cond_seq = build_cond_seq_from_actions(actions)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latents.shape, device=device, dtype=torch.float32, generator=generator)
    if cond_seq.shape[1] != total_latents:
        raise RuntimeError('Action and latent timelines differ')
    return {'latents': latents, 'cond_seq': cond_seq, 'noise': noise}


def tensor_sha256(tensor):
    """Diagnostic helper: transfers/synchronizes; call strictly outside timing."""
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def input_hashes(inputs):
    return {name: tensor_sha256(value) for name, value in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, nargs='+', default=list(DEFAULT_EPISODES))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-legacy', action='store_true')
    parser.add_argument('--vae-checkpoint', default='/data2/jian/checkpoints/miniworld/Wan2.2_VAE.pth')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [episode_manifest(ep) for ep in args.episodes]
    (args.output_dir/'input_manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')
    if args.validate_legacy:
        from miniworld.vae.codec import load_wan22_vae
        data = load_episode_cpu(2352, include_gt=True)
        archived = torch.load(LEGACY_ROOT/'prepared_inputs.pt', map_location='cpu', weights_only=True)
        archived_rgb = np.load(LEGACY_ROOT/'gt_rgb.npy', mmap_mode='r')
        vae = load_wan22_vae({'vae_checkpoint': args.vae_checkpoint})
        inputs = prepare_inputs(data, vae, 'cuda')
        result = {
            'rgb_bitwise_equal': bool(np.array_equal(data['gt_rgb'], archived_rgb)),
            'cond_seq_bitwise_equal': bool(torch.equal(inputs['cond_seq'].cpu(), archived['cond_seq'])),
            'noise_bitwise_equal': bool(torch.equal(inputs['noise'].cpu(), archived['noise'])),
            'first_latent_bitwise_equal': bool(torch.equal(inputs['latents'][:, :, :1].cpu(), archived['latents'][:, :, :1])),
            'first_latent_max_abs': float((inputs['latents'][:, :, :1].cpu()-archived['latents'][:, :, :1]).abs().max()),
            'all_future_latent_placeholders_zero': bool(torch.count_nonzero(inputs['latents'][:, :, 1:]) == 0),
            'input_hashes': input_hashes(inputs),
            'noise_generator': 'CUDA isolated generator seeded 20263368',
        }
        (args.output_dir/'legacy_parity.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
