#!/usr/bin/env python3
"""Compare lossless RGB outputs from the episode 002352 attention A/B run.

Metrics use pre-MP4 uint8 RGB and exclude conditioning frame 0 from averages.
The primary A/B comparison uses matched runs on each GPU; repeated runs of the
same backend on the other GPU expose reproducibility differences separately.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Keep this analysis away from the generation GPUs and bound CPU parallelism.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

import numpy as np


DEFAULT_ROOT = Path(
    "/data2/jian/output/Miniworld/DROID_long_horizon/"
    "chunk_flash_attention/episode_002352"
)
RUNS = ("gpu4_baseline", "gpu4_chunk_flash", "gpu5_chunk_flash", "gpu5_baseline")
METRICS = ("psnr_db", "ssim", "lpips_alex")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No records for {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def pixel_metrics(pair):
    from skimage.metrics import structural_similarity

    reference, prediction = pair
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference**2))
    return {
        "psnr_db": float("inf") if mse == 0 else 10 * math.log10(255**2 / mse),
        "ssim": float(
            structural_similarity(
                reference,
                prediction,
                data_range=255,
                channel_axis=-1,
                gaussian_weights=True,
                sigma=1.5,
                win_size=11,
                use_sample_covariance=False,
            )
        ),
        "mae_uint8": float(np.mean(np.abs(difference))),
        "max_abs_uint8": int(np.max(np.abs(difference))),
        "identical_pixels_fraction": float(np.mean(difference == 0)),
    }


def measure_pair(reference, prediction, model, device, batch_size):
    import torch

    # SSIM is evaluated on RGB channels independently by skimage, then averaged.
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(pixel_metrics, zip(reference, prediction)))
    with torch.inference_mode():
        for start in range(0, len(reference), batch_size):
            end = min(start + batch_size, len(reference))
            a = torch.from_numpy(np.array(reference[start:end], copy=True))
            b = torch.from_numpy(np.array(prediction[start:end], copy=True))
            a = a.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 127.5 - 1
            b = b.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 127.5 - 1
            # LPIPS-Alex v0.1 runs in FP32; do not inherit generation autocast.
            with torch.autocast(device_type=device.type, enabled=False):
                values = model(a, b).flatten().cpu().tolist()
            for frame, value in enumerate(values, start):
                rows[frame]["lpips_alex"] = float(value)
    return [{"frame": index, **row} for index, row in enumerate(rows)]


def summarize(rows, first_frame, last_frame):
    selected = [row for row in rows if first_frame <= row["frame"] <= last_frame]
    result = {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "num_frames": len(selected),
    }
    for metric in (*METRICS, "mae_uint8", "identical_pixels_fraction"):
        result[metric] = float(np.mean([row[metric] for row in selected]))
    result["max_abs_uint8"] = max(row["max_abs_uint8"] for row in selected)
    return result


def analyze(root, batch_size, device_name):
    import lpips
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    device = torch.device(device_name)
    paths = {"gt": root / "gt_rgb.npy", **{run: root / f"{run}_rgb.npy" for run in RUNS}}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Generation outputs are not complete: " + ", ".join(missing))
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    expected = arrays["gt"].shape
    if len(expected) != 4 or expected[-1] != 3 or expected[0] < 2:
        raise ValueError(f"Expected T x H x W x RGB, got {expected}")
    for name, value in arrays.items():
        if value.shape != expected or value.dtype != np.uint8:
            raise ValueError(f"Invalid shape/dtype for {name}: {value.shape}, {value.dtype}")
    model = lpips.LPIPS(net="alex", version="0.1").eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    comparisons = [(run, "gt", run, "vs_gt") for run in RUNS]
    comparisons += [
        ("gpu4_baseline_vs_chunk_flash", "gpu4_baseline", "gpu4_chunk_flash", "backend_difference"),
        ("gpu5_baseline_vs_chunk_flash", "gpu5_baseline", "gpu5_chunk_flash", "backend_difference"),
        ("baseline_gpu4_vs_gpu5", "gpu4_baseline", "gpu5_baseline", "same_backend_reproducibility"),
        ("chunk_flash_gpu4_vs_gpu5", "gpu4_chunk_flash", "gpu5_chunk_flash", "same_backend_reproducibility"),
    ]
    frame_rows = []
    summary_rows = []
    comparisons_json = {}
    nframes = expected[0]
    # Quarter boundaries are relative to generated frames 1..T-1, not seed 0.
    last_quarter_start = 1 + 3 * (nframes - 1) // 4
    for name, reference_name, prediction_name, kind in comparisons:
        start = time.perf_counter()
        print(f"Measuring {name} ({nframes} RGB frames)", flush=True)
        rows = measure_pair(arrays[reference_name], arrays[prediction_name], model, device, batch_size)
        for row in rows:
            frame_rows.append({"comparison": name, "kind": kind, **row})
        comparison = {
            "reference": reference_name,
            "prediction": prediction_name,
            "kind": kind,
        }
        for interval, first in (("all_generated", 1), ("last_quarter", last_quarter_start)):
            values = summarize(rows, first, nframes - 1)
            comparison[interval] = values
            summary_rows.append({"comparison": name, "kind": kind, "interval": interval, **values})
        comparison["analysis_seconds"] = time.perf_counter() - start
        comparisons_json[name] = comparison
        print(json.dumps(json_safe(comparison["all_generated"])), flush=True)
        # Checkpoint each completed comparison so an interrupted analysis is inspectable.
        write_csv(root / "quality_per_frame.csv", frame_rows)
        write_csv(root / "quality_summary.csv", summary_rows)
    backend_rows = []
    for backend in ("baseline", "chunk_flash"):
        for interval in ("all_generated", "last_quarter"):
            matched = [comparisons_json[f"gpu{gpu}_{backend}"][interval] for gpu in (4, 5)]
            backend_rows.append({
                "backend": backend,
                "interval": interval,
                "num_runs": 2,
                "first_frame": matched[0]["first_frame"],
                "last_frame": matched[0]["last_frame"],
                "num_frames_per_run": matched[0]["num_frames"],
                **{metric: float(np.mean([row[metric] for row in matched])) for metric in METRICS},
            })
    write_csv(root / "quality_by_backend.csv", backend_rows)
    payload = {
        "episode": "002352",
        "shape_thwc": list(expected),
        "source": "lossless uint8 RGB before MP4 compression",
        "seed_frame_excluded_from_averages": 0,
        "psnr": "per-frame RGB PSNR with data_range=255, arithmetic mean over frames",
        "ssim": "RGB channel mean; gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False, data_range=255",
        "lpips": "AlexNet v0.1; FP32; uint8 / 127.5 - 1; no autocast",
        "lpips_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "comparisons": comparisons_json,
        "backend_means": backend_rows,
        "note": "Same seed and episode; the two runs per backend are device/order repeats, not independent episodes or seeds.",
    }
    (root / "quality_analysis.json").write_text(json.dumps(json_safe(payload), indent=2) + "\n")


def plot_quality(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (root / "quality_per_frame.csv").open() as file:
        rows = list(csv.DictReader(file))
    runs = {
        run: [row for row in rows if row["comparison"] == run and int(row["frame"]) > 0]
        for run in RUNS
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for axis, metric, ylabel in zip(axes, METRICS, ("PSNR (dB) ↑", "SSIM ↑", "LPIPS-Alex ↓")):
        for backend, color, label in (
            ("baseline", "#2878b5", "Original masked SDPA"),
            ("chunk_flash", "#d55e00", "Chunk FlashAttention"),
        ):
            frames = np.array([int(row["frame"]) for row in runs[f"gpu4_{backend}"]])
            values = np.array([[float(row[metric]) for row in runs[f"gpu{gpu}_{backend}"]] for gpu in (4, 5)])
            axis.fill_between(frames, values.min(axis=0), values.max(axis=0), color=color, alpha=0.14, linewidth=0)
            axis.plot(frames, values.mean(axis=0), color=color, linewidth=1.25, label=label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="best")
    axes[-1].set_xlabel("RGB frame index (conditioning frame 0 excluded)")
    axes[-1].set_xlim(1, int(frames[-1]))
    fig.suptitle("Episode 002352 | Autoregressive generated history | Quality versus ground truth", fontsize=14)
    fig.text(0.075, 0.018, "Unsmoothed curves: mean of matched GPU 4 / GPU 5 runs; shading: range across repeats.\nMetrics use lossless uint8 RGB before MP4 encoding. Sink, history capacity and sampling settings are unchanged.", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for extension in ("png", "pdf"):
        fig.savefig(root / f"quality_vs_frame.{extension}", dpi=180)
    plt.close(fig)


def font(size):
    from PIL import ImageFont

    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size=size) if Path(path).exists() else ImageFont.load_default()


def make_visuals(root):
    import av
    from PIL import Image, ImageDraw

    # A single matched GPU pair is shown throughout; never mix frames from runs.
    arrays = [np.load(root / name, mmap_mode="r") for name in (
        "gt_rgb.npy", "gpu4_baseline_rgb.npy", "gpu4_chunk_flash_rgb.npy"
    )]
    nframes, height, width, _ = arrays[0].shape
    titles = ("Ground truth", "Original masked SDPA", "Chunk FlashAttention")
    frames = [min(index, nframes - 1) for index in (64, 128, 256, 384, 496)]
    label_height, top_height, left_width = 30, 54, 84
    sheet = Image.new("RGB", (left_width + 3 * width, top_height + len(frames) * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 7), "Episode 002352 | Matched GPU 4 runs | Autoregressive generated history", fill="black", font=font(18))
    for column, title in enumerate(titles):
        draw.text((left_width + column * width + 8, 32), title, fill="black", font=font(16))
    for row, frame in enumerate(frames):
        y = top_height + row * (height + label_height)
        draw.text((8, y + 10), f"Frame\n{frame}", fill="black", font=font(16))
        for column, array in enumerate(arrays):
            sheet.paste(Image.fromarray(array[frame]), (left_width + column * width, y))
    sheet.save(root / "visual_comparison.png")
    # Keep full-size individual outputs available in addition to the comparison.
    for filename, array in (("baseline.mp4", arrays[1]), ("chunk_flash.mp4", arrays[2])):
        with av.open(str(root / filename), mode="w") as container:
            stream = container.add_stream("libx264", rate=15)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            for rgb in array:
                frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    video_path = root / "gt_baseline_chunk_flash_triptych.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=15)
        stream.width = 3 * width
        stream.height = height + 56
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for index in range(nframes):
            canvas = Image.new("RGB", (3 * width, height + 56), "white")
            draw = ImageDraw.Draw(canvas)
            for column, (title, array) in enumerate(zip(titles, arrays)):
                draw.text((column * width + 8, 5), title, fill="black", font=font(16))
                canvas.paste(Image.fromarray(array[index]), (column * width, 56))
            draw.text((8, 29), f"Episode 002352 | RGB frame {index:03d} / {nframes - 1} | Matched GPU 4 runs", fill="black", font=font(14))
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    print(f"Wrote contact sheet and triptych video: {video_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--visuals-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_quality(args.root)
        return
    if not args.visuals_only:
        analyze(args.root, args.batch_size, args.device)
        if importlib.util.find_spec("matplotlib") is not None:
            plot_quality(args.root)
        else:
            # The inference environment intentionally has no plotting dependency.
            subprocess.run([
                "/home/jian/anaconda3/bin/python3", str(Path(__file__).resolve()),
                "--root", str(args.root), "--plot-only",
            ], check=True)
    make_visuals(args.root)


if __name__ == "__main__":
    main()
