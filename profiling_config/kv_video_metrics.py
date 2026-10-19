"""Full-frame RGB PSNR, Gaussian SSIM, and official LPIPS-Alex v0.1.

Inputs are float RGB in [0, 1], before lossy video encoding. The default
excludes RGB frame zero, the conditioning image. No frames are subsampled,
resized, quantized, or silently clamped. Metric work belongs outside generation
timing. The callable returns JSON-safe per-frame vectors and aggregate scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F


LPIPS_SOURCE_COMMIT = "082bb24f84c091ea94de2867d34c4544f68e0963"
LPIPS_HEAD_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _frames(video):
    """Accept TCHW or the decoder's 1CTHW layout without copying all frames."""
    video = torch.as_tensor(video).detach()
    if video.ndim == 5 and video.shape[0] == 1 and video.shape[1] == 3:
        video = video[0].permute(1, 0, 2, 3)
    if video.ndim != 4 or video.shape[1] != 3 or video.shape[0] < 1:
        raise ValueError("expected float RGB [T,3,H,W] or [1,3,T,H,W]")
    if not video.is_floating_point():
        raise ValueError("RGB must already be floating point in [0,1]")
    if min(video.shape[-2:]) < 64:
        raise ValueError("LPIPS-Alex evaluation requires image dimensions at least 64; no resizing is performed")
    if not bool(torch.isfinite(video).all()):
        raise ValueError("RGB contains nonfinite values")
    low, high = torch.aminmax(video)
    if float(low) < 0 or float(high) > 1:
        raise ValueError(f"RGB must be in [0,1], got [{float(low)}, {float(high)}]")
    return video


def gaussian_ssim(reference, actual):
    """Per-image RGB SSIM: Gaussian11, sigma1.5, population covariance.

    The mean uses valid window centres (five pixels cropped on each edge),
    matching skimage's gaussian_weights=True, sigma=1.5, win_size=11,
    use_sample_covariance=False, channel_axis=2, data_range=1.0.
    """
    if reference.shape != actual.shape or reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("SSIM expects matching [N,3,H,W]")
    if min(reference.shape[-2:]) < 11:
        raise ValueError("SSIM requires an 11x11 window")
    x, y = reference.float(), actual.float()
    coord = torch.arange(11, device=x.device, dtype=torch.float64) - 5
    kernel = torch.exp(-coord.square() / (2 * 1.5 ** 2))
    kernel = (kernel / kernel.sum()).float()
    moments = torch.cat((x, y, x.square(), y.square(), x * y), dim=1)
    horizontal = kernel.view(1, 1, 1, 11).expand(15, 1, 1, 11)
    vertical = kernel.view(1, 1, 11, 1).expand(15, 1, 11, 1)
    moments = F.conv2d(F.conv2d(moments, horizontal, groups=15), vertical, groups=15)
    ux, uy, ex2, ey2, exy = moments.chunk(5, dim=1)
    vx, vy, vxy = ex2 - ux.square(), ey2 - uy.square(), exy - ux * uy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)
             / ((ux.square() + uy.square() + c1) * (vx + vy + c2)))
    return score.mean(dim=(1, 2, 3))


class VideoMetricCalculator:
    """Reusable FP32 LPIPS evaluator with bounded GPU frame batches.

    LPIPS is always the official pretrained AlexNet v0.1 calibrated metric.
    Both checkpoint hashes and loaded parameter values are verified. Failure
    to obtain or load those weights raises an error; there is no random-weight
    fallback. Construct after generation, or keep this module's allocation
    outside generation memory/timing measurements.
    """

    def __init__(self, device="cuda:0", batch_size=8, lpips_net="alex"):
        if lpips_net != "alex":
            raise ValueError("this evaluation fixes LPIPS to official AlexNet v0.1")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device, self.batch_size = torch.device(device), batch_size
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for metrics but unavailable")
        started = time.perf_counter()
        import lpips
        import torchvision

        # Model constructors initialize temporary parameters before pretrained
        # loading. Preserve CPU RNG so metric setup cannot perturb later runs.
        with torch.random.fork_rng(devices=[]):
            self.lpips = lpips.LPIPS(net="alex", version="0.1", pretrained=True,
                                     pnet_rand=False, pnet_tune=False,
                                     lpips=True, eval_mode=True, verbose=False)
        head_path = Path(lpips.__file__).resolve().parent / "weights/v0.1/alex.pth"
        backbone_path = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
        head_hash, backbone_hash = _sha256(head_path), _sha256(backbone_path)
        if head_hash != LPIPS_HEAD_SHA256 or backbone_hash != ALEXNET_SHA256:
            raise RuntimeError("LPIPS/AlexNet checkpoint checksum differs from verified official weights")
        heads = torch.load(head_path, map_location="cpu", weights_only=True)
        loaded = self.lpips.state_dict()
        if len(heads) != 5 or any(key not in loaded or not torch.equal(value, loaded[key])
                                  for key, value in heads.items()):
            raise RuntimeError("official LPIPS calibration weights were not completely loaded")
        backbone = torch.load(backbone_path, map_location="cpu", weights_only=True)
        trunk_count = 0
        for name, value in self.lpips.net.state_dict().items():
            feature_key = "features." + name.split(".", 1)[1]
            if feature_key not in backbone or not torch.equal(value, backbone[feature_key]):
                raise RuntimeError(f"LPIPS backbone differs from pretrained AlexNet at {name}")
            trunk_count += 1
        if trunk_count != 10:
            raise RuntimeError(f"unexpected AlexNet feature parameter count: {trunk_count}")
        del backbone, loaded, heads
        self.lpips = self.lpips.float().eval().requires_grad_(False).to(self.device)
        self.provenance = {
            "lpips": {
                "implementation": "richzhang/PerceptualSimilarity (official lpips package)",
                "source_url": f"https://github.com/richzhang/PerceptualSimilarity/tree/{LPIPS_SOURCE_COMMIT}",
                "package_version": importlib.metadata.version("lpips"),
                "network": "alex", "metric_version": "0.1", "calibrated": True,
                "pretrained_backbone": True, "random_weights": False,
                "input_range": [-1.0, 1.0], "dtype": "float32", "eval_mode": True,
                "calibration_path": str(head_path), "calibration_sha256": head_hash,
                "backbone_path": str(backbone_path), "backbone_sha256": backbone_hash,
                "backbone_official_url": ALEXNET_URL,
                "loaded_parameters_verified": True,
            },
            "ssim": {
                "window": 11, "sigma": 1.5, "gaussian_weights": True,
                "use_sample_covariance": False, "data_range": 1.0,
                "K1": 0.01, "K2": 0.03, "channels": "RGB arithmetic mean",
                "border": "valid window centres; crop five pixels per edge",
                "implementation": "separable FP32 PyTorch Gaussian convolution",
                "reference_url": "https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity",
            },
            "psnr": {
                "data_range": 1.0, "mse_reduction_dtype": "float64",
                "formula": "-10 log10(mean squared RGB pixel error)",
                "epsilon_or_cap": None,
                "json_infinity": "null PSNR with explicit positive-infinity flag when MSE is exactly zero",
            },
            "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "device": str(self.device), "frame_batch_size": batch_size,
            "setup_wall_seconds": time.perf_counter() - started,
            "metric_time_excluded_from_generation": True,
        }

    @torch.inference_mode()
    def __call__(self, reference, actual, *, exclude_first_frames=1, frame_indices=None):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        reference, actual = _frames(reference), _frames(actual)
        if reference.shape != actual.shape:
            raise ValueError("reference and actual frames must have identical shapes and timestamps; no resizing/truncation")
        if not isinstance(exclude_first_frames, int) or not 0 <= exclude_first_frames < reference.shape[0]:
            raise ValueError("exclude_first_frames must leave at least one evaluated frame")
        total = reference.shape[0]
        frame_indices = list(range(total)) if frame_indices is None else list(frame_indices)
        if len(frame_indices) != total:
            raise ValueError("frame_indices must identify every supplied reference/actual frame")
        if any(not isinstance(index, int) for index in frame_indices):
            raise ValueError("frame_indices must contain Python integer RGB frame identifiers")
        records = {"rgb_frame": frame_indices[exclude_first_frames:],
                   "mse_rgb": [], "psnr_db": [], "psnr_is_positive_infinity": [],
                   "ssim": [], "lpips_alex": []}
        with torch.autocast(self.device.type, enabled=False), torch.backends.cudnn.flags(allow_tf32=False):
            for start in range(exclude_first_frames, total, self.batch_size):
                stop = min(start + self.batch_size, total)
                ref = reference[start:stop].to(self.device, dtype=torch.float32)
                pred = actual[start:stop].to(self.device, dtype=torch.float32)
                mse = (ref.double() - pred.double()).square().mean(dim=(1, 2, 3))
                ssim = gaussian_ssim(ref, pred)
                perceptual = self.lpips(ref * 2 - 1, pred * 2 - 1, normalize=False).flatten()
                if not bool(torch.isfinite(ssim).all() & torch.isfinite(perceptual).all()):
                    raise RuntimeError("nonfinite SSIM or LPIPS result")
                mse_values = mse.cpu().tolist()
                records["mse_rgb"].extend(mse_values)
                records["psnr_db"].extend(None if value == 0 else -10 * math.log10(value) for value in mse_values)
                records["psnr_is_positive_infinity"].extend(value == 0 for value in mse_values)
                records["ssim"].extend(ssim.cpu().tolist())
                records["lpips_alex"].extend(perceptual.cpu().tolist())
        count = len(records["mse_rgb"])
        global_mse = math.fsum(records["mse_rgb"]) / count
        finite_psnr = [value for value in records["psnr_db"] if value is not None]
        infinite_count = count - len(finite_psnr)
        summary = {
            "frame_count": count, "mse_global": global_mse,
            "psnr_db_global": None if global_mse == 0 else -10 * math.log10(global_mse),
            "psnr_db_global_is_positive_infinity": global_mse == 0,
            "psnr_db_mean": None if infinite_count else math.fsum(finite_psnr) / count,
            "psnr_db_mean_is_positive_infinity": infinite_count > 0,
            "psnr_db_finite_frame_mean": math.fsum(finite_psnr) / len(finite_psnr) if finite_psnr else None,
            "psnr_positive_infinite_frames": infinite_count,
            "ssim_mean": math.fsum(records["ssim"]) / count,
            "lpips_alex_mean": math.fsum(records["lpips_alex"]) / count,
        }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "summary": summary, "per_frame": records,
            "evaluation": {"provided_frames": total, "excluded_first_frames": exclude_first_frames,
                           "evaluated_frames": count, "height": reference.shape[-2], "width": reference.shape[-1],
                           "source": "unencoded float RGB [0,1]", "subsampling": None,
                           "reference_alignment": "caller-supplied identical frame coordinates",
                           "wall_seconds": time.perf_counter() - started,
                           "included_in_generation_timing": False},
            "provenance": self.provenance,
        }


def self_test(device="cpu"):
    """Real pretrained identity/perturbation checks plus independent SSIM oracle."""
    import numpy as np
    from skimage.metrics import structural_similarity

    generator = torch.Generator().manual_seed(1926)
    reference = torch.rand(4, 3, 64, 80, generator=generator)
    actual = (reference + .06 * torch.randn(reference.shape, generator=generator)).clamp(0, 1)
    state = torch.random.get_rng_state().clone()
    calculator = VideoMetricCalculator(device=device, batch_size=2)
    assert torch.equal(state, torch.random.get_rng_state()), "metric initialization changed CPU RNG"
    identity = calculator(reference, reference)
    changed = calculator(reference, actual)
    assert identity["summary"]["frame_count"] == 3
    assert identity["summary"]["mse_global"] == 0
    assert identity["summary"]["psnr_db_global_is_positive_infinity"]
    assert abs(identity["summary"]["ssim_mean"] - 1) < 1e-6
    assert abs(identity["summary"]["lpips_alex_mean"]) < 1e-8
    assert changed["summary"]["psnr_db_global"] > 10
    assert changed["summary"]["ssim_mean"] < .999
    assert changed["summary"]["lpips_alex_mean"] > 0
    expected = [structural_similarity(x.numpy().transpose(1, 2, 0), y.numpy().transpose(1, 2, 0),
                                      channel_axis=2, data_range=1., gaussian_weights=True,
                                      sigma=1.5, use_sample_covariance=False, win_size=11)
                for x, y in zip(reference[1:], actual[1:])]
    np.testing.assert_allclose(changed["per_frame"]["ssim"], expected, atol=3e-6, rtol=3e-6)
    expected_mse = (reference[1:].double()-actual[1:].double()).square().mean().item()
    assert abs(changed["summary"]["mse_global"] - expected_mse) < 1e-15
    assert changed["per_frame"]["rgb_frame"] == [1, 2, 3]
    json.dumps(identity, allow_nan=False)
    return {"status": "passed", "device": device,
            "identity": identity["summary"], "perturbation": changed["summary"],
            "ssim_max_abs_error_vs_skimage": max(abs(a-float(b)) for a,b in zip(changed["per_frame"]["ssim"], expected)),
            "perturbation_metric_wall_seconds": changed["evaluation"]["wall_seconds"],
            "provenance": calculator.provenance}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; import VideoMetricCalculator for full-video evaluation")
    result = self_test(args.device)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)
