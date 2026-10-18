"""Reproduce episode 002352 with the archived 500-frame sampling settings.

Prepare once, then run baseline/chunk_flash in opposite orders on two GPUs.
Each pair runs sequentially on the same GPU, with no diagnostic hooks enabled.
End-to-end timing includes video/action loading, GPU transfer, VAE encoding,
conditioning, sampling/cache management, and streaming VAE decoding. Model
loading, warmup, metric computation and output encoding are excluded.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from miniworld import sample
from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.data.droid import LeRobotActionDataset
from miniworld.vae.codec import load_wan22_vae, vae_encode, StreamingVAEDecoder

OUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')
OLD = OUT.parents[1] / '500_frames'
SEED = 20263368
EPISODE = 2352
TOTAL_LATENTS = 125
TOTAL_RGB = 497


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def archived_args():
    return argparse.Namespace(**json.loads((OLD / 'worker_5_environment.json').read_text())['args'])


def load_model():
    args = archived_args()
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    weights = raw.get('ema_model', raw.get('model', raw))
    model = sample.build_denoiser(args)
    model.load_state_dict(weights, strict=True)
    model = model.cuda().eval()
    return model, args


def make_dataset():
    dataset = LeRobotActionDataset(
        '/data2/jian/benchmark/droid', num_frames=TOTAL_RGB, frame_interval=1,
        resize_hw=(240, 320), randomize=False, color_aug=False, max_keep=1,
    )
    entry = json.loads((OLD / 'episode_002352_summary.json').read_text())
    dataset._parquet_path = lambda _: Path(entry['parquet'])
    return dataset, entry


def decode_source(dataset, entry):
    video, frame_ids = dataset._decode_clip(Path(entry['source']))
    assert frame_ids == list(range(TOTAL_RGB))
    actions = dataset._load_actions(EPISODE, frame_ids).unsqueeze(0).cuda()
    return video, actions


def encode_inputs(vae, video, actions):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        latents = vae_encode(vae, video.unsqueeze(0).cuda().permute(0, 4, 1, 2, 3).contiguous())
        assert tuple(latents.shape) == (1, 48, TOTAL_LATENTS, 15, 20)
        cond = build_cond_seq_for_batch(
            cfg=ConditioningConfig(False, True, 15), poses=None, actions=actions,
            t_latent=TOTAL_LATENTS, h_lat=15, w_lat=20,
        )
    # Ordinary autoregression receives only the initial observation. Keeping
    # later GT latents zero makes the absence of teacher-forcing explicit.
    latents[:, :, 1:] = 0
    return dict(latents=latents, cond_seq=cond)


def load_prepared(device='cuda'):
    data = torch.load(OUT / 'prepared_inputs.pt', map_location='cpu', weights_only=True)
    return {key: value.to(device) for key, value in data.items()}


def set_backend(model, backend):
    assert backend in ('baseline', 'chunk_flash')
    if not hasattr(model.net, '_benchmark_original_forward'):
        model.net._benchmark_original_forward = model.net.forward_with_cache
    original = model.net._benchmark_original_forward
    model.net.forward_with_cache = functools.partial(original, use_chunk_flash_attn=backend == 'chunk_flash')


def generate(model, prepared, decoder=None):
    return model.generate_eval_latents_streaming(
        prepared['latents'], prepared['cond_seq'], total_len=TOTAL_LATENTS,
        history_len=1, max_cache_chunks=24, inflight_chunks=8, sink_frames=1,
        stream_decoder=decoder, collect_stream_timing=False, noise=prepared['noise'],
        # This historical benchmark isolates the attention backend. Keep its
        # original KV implementation even when the sampler default evolves.
        use_resident_kv_cache=False,
    )


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    args = archived_args()
    dataset, entry = make_dataset()
    vae = load_wan22_vae(args)
    video, actions = decode_source(dataset, entry)
    data = encode_inputs(vae, video, actions)
    torch.manual_seed(SEED)
    data['noise'] = torch.randn_like(data['latents'])
    np.save(OUT / 'gt_rgb.npy', ((video + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy())
    temporary = OUT / 'prepared_inputs.pt.tmp'
    torch.save({key: value.cpu() for key, value in data.items()}, temporary)
    temporary.replace(OUT / 'prepared_inputs.pt')
    dump(OUT / 'experiment_manifest.json', dict(
        episode=EPISODE, source=entry['source'], actions=entry['parquet'], seed=SEED,
        original_rgb_frames=500, processed_rgb_frames=TOTAL_RGB, latent_frames=TOTAL_LATENTS,
        args=vars(args), generation_total_len_override=TOTAL_LATENTS,
        mode='normal generated-history autoregression; GT initial latent only',
        history_definition='sliding history only, excludes sink and inflight chunks',
        original_cache_capacity_including_sink_latents=48,
        sink_latents=1, saturated_history_chunks_excluding_sink=23.5,
        inflight_chunks=8, latent_per_chunk=2, effective_sampling_steps=40,
        timing_scope='warm pipeline: source decode/actions + transfers + VAE encode + conditioning + sampling with streaming VAE decode',
        timing_exclusions=['model loading', 'warmup', 'metric calculation', 'video/file output'],
        warmup='VAE encode + baseline cache prefill to 48 latents + both backends at current lengths 16/2/1 + streaming VAE decode',
        collect_stream_timing=False,
        archived_protocol_note='Old timing used per-chunk diagnostic synchronization and a 64-latent warmup; this paired benchmark warms individual operations and disables diagnostic timing equally for both backends.',
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        production_source_sha256=hashlib.sha256((ROOT / 'miniworld/miniworld.py').read_bytes()).hexdigest(),
        noise_sha256=hashlib.sha256(data['noise'].cpu().numpy().tobytes()).hexdigest(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
    ))
    print('PREPARED', OUT, flush=True)


def warmup(model, vae, data):
    # Warm both kernels, the longest tested KV shape, and VAE streaming decode.
    # These forwards are not scored and do not change sampler configuration.
    net = model.net
    past = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for offset in (0, 16, 32):
            _, new = net.forward_with_cache(
                data['noise'][:, :, :16], torch.zeros(1, 16, device='cuda'), data['cond_seq'][:, :16],
                past_kv_list=past, current_position_offset=offset,
                return_kv=True, chunk_size=2, use_chunk_flash_attn=False,
            )
            past = model._append_kv_cache([None] * net.depth if past is None else past, new)
        for fast in (False, True):
            for count in (16, 2, 1):
                net.forward_with_cache(
                    data['noise'][:, :, :count], torch.ones(1, count, device='cuda'),
                    data['cond_seq'][:, :count], past_kv_list=past,
                    current_position_offset=48, return_kv=count < 16,
                    chunk_size=2, use_chunk_flash_attn=fast,
                )
        StreamingVAEDecoder(vae).decode_all(data['latents'][:, :, :3])
    torch.cuda.synchronize()
    del past, new


def run_pair(order, label):
    model, args = load_model()
    vae = load_wan22_vae(args)
    dataset, entry = make_dataset()
    fixed = load_prepared()
    # Encode once before timing to warm the identical input pipeline as well.
    video, actions = decode_source(dataset, entry)
    encode_inputs(vae, video, actions)
    del video, actions
    warmup(model, vae, fixed)
    for backend in order:
        run_id = f'{label}_{backend}'
        set_backend(model, backend)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        dump(OUT / f'{label}_state.json', dict(phase='running', backend=backend, time=time.time()))
        started = time.perf_counter()
        video, actions = decode_source(dataset, entry)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        data = encode_inputs(vae, video, actions)
        data['noise'] = fixed['noise']
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latents, rgb = generate(model, data, StreamingVAEDecoder(vae))
        torch.cuda.synchronize()
        ended = time.perf_counter()
        result = dict(
            run_id=run_id, backend=backend, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            order=order, seed=SEED, source_load_transfer_sec=loaded-started,
            vae_encode_condition_sec=encoded-loaded, generation_stream_decode_sec=ended-encoded,
            end_to_end_sec=ended-started, generated_rgb_fps=496/(ended-started),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            sampler_meta=model.last_eval_meta.copy(),
            prepared_latents_identical=bool(torch.equal(data['latents'], fixed['latents'])),
            conditioning_identical=bool(torch.equal(data['cond_seq'], fixed['cond_seq'])),
        )
        assert result['prepared_latents_identical'] and result['conditioning_identical']
        assert result['sampler_meta']['effective_steps'] == 40
        pixels = ((rgb[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        assert pixels.shape == (TOTAL_RGB, 240, 320, 3)
        np.save(OUT / f'{run_id}_rgb.npy', pixels)
        torch.save(latents.cpu(), OUT / f'{run_id}_latents.pt')
        dump(OUT / f'{run_id}_timing.json', result)
        print('COMPLETED', json.dumps(result), flush=True)
        del video, actions, data, latents, rgb, pixels
    dump(OUT / f'{label}_state.json', dict(phase='complete', time=time.time()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--order', nargs='+', choices=['baseline', 'chunk_flash'])
    parser.add_argument('--label')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.prepare:
        prepare()
    else:
        if args.order is None or args.label is None:
            parser.error('provide --prepare or both --order and --label')
        run_pair(args.order, args.label)


if __name__ == '__main__':
    main()
