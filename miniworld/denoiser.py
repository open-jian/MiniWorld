from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from miniworld.vae.codec import print0 as _print0


class IncrementalTimesteps:
    """AR-Diffusion style combinatorial timestep sampler.
    """

    def __init__(self, F: int, T: int):
        self.F = F
        self.T = T

        mat = torch.zeros((T, F), dtype=torch.float64)
        for t in range(T):
            mat[t, F - 1] = 1
        for f in range(F - 2, -1, -1):
            mat[T - 1, f] = 1
            for t in range(T - 2, -1, -1):
                mat[t, f] = mat[t + 1, f] + mat[t, f + 1]
        self.mat_s = mat.numpy()

        mat = torch.zeros((T, F), dtype=torch.float64)
        for t in range(T):
            mat[t, 0] = 1
        for f in range(1, F):
            mat[0, f] = 1
            for t in range(1, T):
                mat[t, f] = mat[t - 1, f] + mat[t, f - 1]
        self.mat_e = mat.numpy()

    def sample_stepseq_from_mid(self):
        timesteps = torch.zeros(self.F, dtype=torch.long)
        cur_f = np.random.randint(self.F)
        timesteps[cur_f] = np.random.randint(self.T)

        for f in range(cur_f - 1, -1, -1):
            candidate_weights = self.mat_e[: int(timesteps[f + 1]) + 1, f]
            prob_sequence = candidate_weights / candidate_weights.sum()
            cur_step = np.random.choice(range(0, int(timesteps[f + 1]) + 1), p=prob_sequence)
            timesteps[f] = int(cur_step)

        for f in range(cur_f + 1, self.F):
            candidate_weights = self.mat_s[int(timesteps[f - 1]):, f]
            prob_sequence = candidate_weights / candidate_weights.sum()
            cur_step = np.random.choice(range(int(timesteps[f - 1]), self.T), p=prob_sequence)
            timesteps[f] = int(cur_step)
        return timesteps


class DenoiserConfig:
    def __init__(self, **kwargs):
        self.wm_model: str = "1B"
        self.latent_size: int = 16
        self.latent_channels: int = 48
        self.latent_frames: int = 9

        self.wm_mlp_ratio: float = 4.0
        self.wm_use_qknorm: bool = True
        self.wm_use_checkpoint: bool = True
        self.cond_dim: int = 0
        # When True, y is treated as per-token spatial conditioning
        # ``(B, T, cond_dim, H_lat, W_lat)`` (e.g. ray-encoding for camera
        # pose). When False (default), y is the per-frame ``(B, T, cond_dim)``
        # latent-action condition.
        self.cond_per_token: bool = False

        # Structured action/pose dropout for classifier-free guidance training.
        self.adaln_mode: str = "adaln_lora"
        self.cond_dropout_prob: float = 0.0
        # Route the true first latent frame (seed / initial observation, no
        # preceding action) through the learned null_action (action mode only).
        self.action_null_first: bool = True

        # Long-video finetune / streaming inference metadata.
        # ``trained_num_frames`` defaults to ``latent_frames`` and is saved in the
        # ckpt meta so streaming inference can assert the active window
        # (cache + in-flight) never exceeds it.
        self.trained_num_frames: int = -1  # -1 => fallback to latent_frames at runtime

        # Training timesteps: t = sigmoid(P_mean + P_std * z), z ~ N(0, 1).
        # P_std <= 0 falls back to uniform.
        self.P_mean: float = 0.0
        self.P_std: float = 1.0
        self.timestep_shift: float = -1.0  # -1 = auto from per-chunk token count; >0 = manual override
        self.timestep_baseshift: float = 2.667  # shift at _REF_TOKENS; see Denoiser.__init__

        # sample
        self.num_sampling_steps: int = 50
        self.cfg_scale: float = 1.0
        self.cfg_interval_min: float = 0.1
        self.cfg_interval_max: float = 1.0

        self.df_chunk_size: int = 2
        self.df_train_time_bins: int = 50
        self.df_ardiff_step: int = 1

        # Opt-in, inference-only head routing. Disabled preserves the original
        # sampler and does not instantiate/import any HeadCast runtime.
        self.headcast: bool = False
        self.headcast_tile_height: int = 5
        self.headcast_tile_width: int = 5
        self.headcast_recent_chunks: int = 2
        self.headcast_max_relative_rmse: float = 0.10
        self.headcast_min_cosine: float = 0.95
        self.headcast_query_samples_per_chunk: int = 32

        # Full physically compact cache path. The default takes the unchanged
        # original sampler; runtimes are private to each enabled rollout.
        self.kv_acceleration: str = 'off'
        self.kv_spatial_score_threshold: float = 0.755
        self.kv_calibration_frames: int = 8
        self.kv_max_relative_rmse: float = 0.10
        self.kv_min_cosine: float = 0.95
        self.kv_query_samples_per_chunk: int = 32
        self.kv_dummy_frames: int = 1
        self.kv_recent_frames: int = 4
        self.kv_tile_height: int = 5
        self.kv_tile_width: int = 5
        self.kv_dummy_fraction: float = 0.5
        # Separate Forcing-KV strategy; these options are read only in its mode.
        self.forcing_kv_threshold: float = 0.8
        self.forcing_kv_segments: int = 6
        self.forcing_kv_retention: float = 0.33
        self.forcing_kv_profile: str = ''
        self.forcing_kv_profile_output: str = ''
        self.forcing_kv_model_identity: str = ''

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
    


class Denoiser(nn.Module):
    """World model Denoiser.

    Args:

    Return:
        diffusion loss
    """

    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()

        self.cfg = cfg
        from miniworld.miniworld import MiniWorldModels

        if cfg.wm_model not in MiniWorldModels:
            raise ValueError(
                f"Unknown MiniWorld model {cfg.wm_model!r}. "
                f"Choose one of {sorted(MiniWorldModels)}."
            )
        self.net = MiniWorldModels[cfg.wm_model](
            input_size=cfg.latent_size,
            in_channels=cfg.latent_channels,
            num_frames=cfg.latent_frames,
            mlp_ratio=cfg.wm_mlp_ratio,
            use_qknorm=cfg.wm_use_qknorm,
            use_rope=True,
            use_abs_pos=False,
            use_checkpoint=cfg.wm_use_checkpoint,
            cond_dim=cfg.cond_dim,
            cond_per_token=cfg.cond_per_token,
            adaln_mode=cfg.adaln_mode,
            cond_dropout_prob=cfg.cond_dropout_prob,
            action_null_first=cfg.action_null_first,
        )

        self.trained_num_frames = (
            cfg.trained_num_frames if cfg.trained_num_frames > 0 else cfg.latent_frames
        )

        # SD3-style timestep shift, scaled by the tokens denoised jointly at one
        # noise level (a single chunk). Under diffusion forcing every chunk has
        # its own t, so a longer window must not move the training t distribution.
        latent_size = cfg.latent_size
        if isinstance(latent_size, (tuple, list)):
            h_lat, w_lat = latent_size
        else:
            h_lat = w_lat = int(latent_size)
        n_tokens = int(cfg.df_chunk_size) * h_lat * w_lat
        if cfg.timestep_shift > 0:
            self.timestep_shift = cfg.timestep_shift
        else:
            _REF_TOKENS = 600  # 2 * 15 * 20: chunk_size=2 at 240x320 @16x downsample
            # Nothing derives the 2.667 default; it is the knob for how hard
            # training leans towards high-noise timesteps.
            self.timestep_shift = cfg.timestep_baseshift * (n_tokens / _REF_TOKENS) ** 0.5
        _print0(f"[Denoiser] latent=({cfg.latent_frames}, {h_lat}, {w_lat}), "
                f"chunk_size={cfg.df_chunk_size}, chunk_tokens={n_tokens}, "
                f"timestep_shift={self.timestep_shift:.4f}")
        # Scheme B: when the net is MiniWorld and its internal structured
        # dropout is enabled, CFG uses the model's *learned null* token for the
        # unconditional branch (train + infer), instead of zeroing cond_seq.
        # This keeps the train-time null and infer-time uncond identical.
        self.use_model_null_cfg = cfg.cond_dropout_prob > 0.0
        # Filled by generate_* so callers can report pipeline throughput.
        self.last_eval_meta: Dict[str, object] = {}

        self.steps = cfg.num_sampling_steps
        self.cfg_scale = cfg.cfg_scale
        self.cfg_interval_min = cfg.cfg_interval_min
        self.cfg_interval_max = cfg.cfg_interval_max
        self.df_chunk_size = int(cfg.df_chunk_size)
        self.df_train_time_bins = max(2, int(cfg.df_train_time_bins))
        self.df_ardiff_step = int(cfg.df_ardiff_step)
        if self.df_ardiff_step <= 0:
            raise ValueError("df_ardiff_step must be > 0 for MiniWorld AR-diffusion")
        self.condition_noise_max_t = 0.05
        self.P_mean = float(cfg.P_mean)
        self.P_std = float(cfg.P_std)
        self._df_train_step_samplers: Dict[int, IncrementalTimesteps] = {}

    def _set_last_eval_meta(
        self,
        *,
        path: str,
        total_chunks: int,
        n_ctx_chunks: int,
        num_outer_steps: int,
        effective_steps: Optional[int] = None,
    ) -> None:
        self.last_eval_meta = {
            "path": path,
            "total_chunks": int(total_chunks),
            "n_ctx_chunks": int(n_ctx_chunks),
            "gen_chunks": int(max(0, total_chunks - n_ctx_chunks)),
            "num_outer_steps": int(num_outer_steps),
            "ar_step": int(self.df_ardiff_step),
            "chunk_size": int(self.df_chunk_size),
            "effective_steps": (
                int(effective_steps) if effective_steps is not None else int(self.steps)
            ),
        }

    def _make_uncond(self, cond_seq: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(cond_for_uncond, cond_drop)`` for the CFG unconditional pass.

        When structured dropout is trained, keep the real conditioning tensor
        and force the model's learned null token via ``cond_drop=all-True``.
        """
        if self.use_model_null_cfg:
            b = cond_seq.shape[0]
            return cond_seq, torch.ones(b, dtype=torch.bool, device=cond_seq.device)
        return torch.zeros_like(cond_seq), None

    def drop_cond(self, cond_seq: torch.Tensor) -> torch.Tensor:
        return cond_seq

    def _build_chunk_slices(self, t: int) -> List[slice]:
        if t <= 0:
            raise ValueError(f"t must be positive, got {t}")
        chunk_size = self.df_chunk_size
        assert chunk_size > 0

        chunk_slices: List[slice] = []
        start = 0
        while start < t:
            end = min(t, start + chunk_size)
            chunk_slices.append(slice(start, end))
            start = end
        return chunk_slices

    def _get_df_train_step_sampler(self, num_chunks: int) -> IncrementalTimesteps:
        sampler = self._df_train_step_samplers.get(num_chunks)
        if sampler is None:
            sampler = IncrementalTimesteps(num_chunks, self.df_train_time_bins)
            self._df_train_step_samplers[num_chunks] = sampler
        return sampler

    def _sample_df_chunk_timesteps(self, num_chunks: int, device: torch.device) -> torch.Tensor:
        if num_chunks <= 0:
            return torch.zeros(0, device=device, dtype=torch.long)
        sampler = self._get_df_train_step_sampler(num_chunks)
        sampled = sampler.sample_stepseq_from_mid()
        return sampled.to(device=device, dtype=torch.long)

    def _broadcast_chunk_values_to_frames(
        self,
        chunk_values: torch.Tensor,
        chunk_slices: List[slice],
        t: int,
    ) -> torch.Tensor:
        b = chunk_values.shape[0]
        frame_values = torch.zeros(b, t, device=chunk_values.device, dtype=chunk_values.dtype)
        for chunk_idx, chunk_slice in enumerate(chunk_slices):
            frame_values[:, chunk_slice] = chunk_values[:, chunk_idx].unsqueeze(1)
        return frame_values

    def _build_async_step_index_matrix(
        self,
        total_chunks: int,
        num_steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # for ar diffusion inference
        if total_chunks <= 0:
            step_index = torch.full((1, total_chunks), num_steps, device=device, dtype=torch.long)
            update_mask = torch.zeros((1, total_chunks), device=device, dtype=torch.bool)
            return step_index, update_mask

        ar_step = int(self.df_ardiff_step)
        pre_row = torch.zeros(total_chunks, dtype=torch.long)
        rows: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        while not torch.all(pre_row == num_steps):
            new_row = torch.zeros_like(pre_row)
            for idx in range(total_chunks):
                if idx == 0 or pre_row[idx - 1] == num_steps:
                    new_row[idx] = pre_row[idx] + 1
                else:
                    new_row[idx] = new_row[idx - 1] - ar_step
            new_row = new_row.clamp(0, num_steps)
            masks.append(new_row != pre_row)
            rows.append(new_row.clone())
            pre_row = new_row

        step_index = torch.stack(rows, dim=0).to(device=device)
        update_mask = torch.stack(masks, dim=0).to(device=device)
        return step_index, update_mask

    def _build_chunk_sampling_schedule(
        self,
        total_chunks: int,
        device: torch.device,
        dtype: torch.dtype,
        n_context_chunks: int = 1,
        effective_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # for ar diffusion inference
        num_steps = effective_steps if effective_steps is not None else int(self.steps)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        ts = self.shift_timestep(ts, self.timestep_shift)
        step_index, update_mask = self._build_async_step_index_matrix(
            total_chunks=total_chunks,
            num_steps=num_steps,
            device=device,
        )

        current_lookup = torch.cat([ts[:1], ts[:-1]], dim=0)
        next_lookup = ts
        t_chunk = current_lookup[step_index]
        t_next_chunk = next_lookup[step_index]
        for ci in range(min(n_context_chunks, total_chunks)):
            t_chunk[:, ci] = 0
            t_next_chunk[:, ci] = 0
            update_mask[:, ci] = False
        return t_chunk, t_next_chunk, update_mask

    def _compute_fifo_valid_intervals(
        self,
        update_mask: torch.Tensor,
        total_chunks: int,
        max_chunks_in_window: int,
    ) -> List[Tuple[int, int]]:
        """Compute per-step FIFO window bounds (chunk-level).

        Mirrors AR-Diffusion ``fifoddim.py``'s ``valid_interval`` logic.
        The window starts covering chunks ``[0, max_chunks_in_window)`` and
        slides right by one chunk each time a new chunk at the window
        boundary becomes active (``update_mask`` turns True).

        Returns a list of ``(start_chunk, end_chunk)`` tuples, one per
        outer iteration.
        """
        terminal = min(max_chunks_in_window, total_chunks)
        intervals: List[Tuple[int, int]] = []
        for i in range(update_mask.shape[0]):
            if terminal < total_chunks and bool(update_mask[i, terminal]):
                terminal += 1
            start = max(0, terminal - max_chunks_in_window)
            intervals.append((start, terminal))
        return intervals


    def _build_diffusion_forcing_timesteps(
        self,
        b: int,
        t: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Build per-frame timesteps for diffusion forcing training.

        Clean-context length is sampled per example:
          Mode A (p=0.5): only the first frame is clean
          Mode B (p=0.5): the entire first chunk is clean

        Returns:
            t_frame:      (B, T)
            chunk_slices: list of slices
            chunk_t:      (B, num_chunks)
            clean_mask:   (B, T)  1 on clean context frames, else 0
        """
        chunk_slices = self._build_chunk_slices(t)
        num_chunks = len(chunk_slices)
        chunk_t = torch.zeros(b, num_chunks, device=device, dtype=dtype)
        scale = float(max(self.df_train_time_bins - 1, 1))

        for sample_idx in range(b):
            if num_chunks <= 0:
                continue
            seq1 = self._sample_df_chunk_timesteps(num_chunks, device=device)
            chunk_t[sample_idx, :] = seq1.to(dtype=dtype) / scale

        chunk_t = self.logit_normal_warp(chunk_t)
        chunk_t = self.shift_timestep(chunk_t, self.timestep_shift)
        t_frame = self._broadcast_chunk_values_to_frames(chunk_t, chunk_slices, t)

        clean_mask = torch.zeros(b, t, device=device, dtype=dtype)
        cond_noise = self.sample_condition_t((b,), device=device, dtype=dtype)

        for sample_idx in range(b):
            if num_chunks <= 0:
                continue
            if torch.rand(1).item() < 0.5:
                # Mode A: only first frame is clean
                t_frame[sample_idx, 0] = cond_noise[sample_idx]
                clean_mask[sample_idx, 0] = 1.0
            else:
                # Mode B: entire first chunk is clean
                first_sl = chunk_slices[0]
                t_frame[sample_idx, first_sl] = cond_noise[sample_idx]
                clean_mask[sample_idx, first_sl] = 1.0

        return t_frame, chunk_slices, chunk_t, clean_mask

    def _get_df_action_guidance_scale(self, chunk_t: torch.Tensor) -> torch.Tensor:
        # Apply cfg_scale when chunk_t is inside
        # (cfg_interval_min, cfg_interval_max]; else 1.0.  Upper bound
        # is inclusive so that the first denoising step (chunk_t == 1.0) still
        # receives CFG, matching diffusion-forcing guidance semantics.
        low = self.cfg_interval_min
        high = self.cfg_interval_max
        interval_mask = (chunk_t <= high) & ((low == 0.0) | (chunk_t > low))
        action_scale = torch.where(
            interval_mask,
            torch.full_like(chunk_t, self.cfg_scale),
            torch.ones_like(chunk_t),
        )
        return action_scale

    def logit_normal_warp(self, u: torch.Tensor) -> torch.Tensor:
        """Give the training timesteps a logit-normal density.

        ``u`` is the uniform bin grid from ``IncrementalTimesteps``. The
        logit-normal inverse CDF is monotone, so it reshapes the density without
        disturbing the non-decreasing noise ordering across chunks.
        """
        if self.P_std <= 0.0:
            return u
        z = torch.special.ndtri(u.to(torch.float64))
        return torch.sigmoid(self.P_mean + self.P_std * z).to(dtype=u.dtype)

    @staticmethod
    def shift_timestep(t: torch.Tensor, shift: float) -> torch.Tensor:
        """SD3-style timestep shift: t' = shift*t / (1 + (shift-1)*t).
        Maps [0,1]->[0,1]; shift>1 biases towards higher t (more noise)."""
        if shift == 1.0:
            return t
        return shift * t / (1.0 + (shift - 1.0) * t)

    def sample_condition_t(self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.condition_noise_max_t <= 0.0:
            return torch.zeros(shape, device=device, dtype=dtype)
        return torch.rand(shape, device=device, dtype=dtype) * self.condition_noise_max_t

    def forward_diffusion_forcing(
        self,
        latents: torch.Tensor,
        cond_seq: torch.Tensor,
        history_len: int = 1,
        return_pred: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``history_len`` is accepted for API compatibility with train/sample CLI,
        # but training clean-context length is sampled via Mode A/B (see
        # ``_build_diffusion_forcing_timesteps``). Inference uses ``history_len``
        # in ``generate_eval_latents_streaming``.
        assert int(history_len) > 0, f"history_len must be > 0, got {history_len}"
        assert latents.dim() == 5, f"latents must be (B, C, T, H, W), got {latents.shape}"
        b, _, t, _, _ = latents.shape
        assert cond_seq.shape[0] == b and cond_seq.shape[1] == t, (
            f"cond_seq shape {cond_seq.shape} must match (B, T, D) with B={b}, T={t}"
        )

        cond_seq = self.drop_cond(cond_seq)
        device = latents.device

        t_frame, _, _, clean_mask = self._build_diffusion_forcing_timesteps(
            b=b,
            t=t,
            device=device,
            dtype=latents.dtype,
        )

        noise = torch.randn_like(latents)
        v_target = latents - noise

        t_view = t_frame.view(b, 1, t, 1, 1)
        z = (1.0 - t_view) * latents + t_view * noise
        v_pred = self.net(
            z,
            t_frame,
            cond_seq,
            temporal_causal=True,
            chunk_size=self.df_chunk_size,
        )

        # --- v_loss (per-frame, excluding clean context) ---
        diff = (v_target - v_pred) ** 2
        diff = diff.mean(dim=(1, 3, 4))  # (B, T)
        loss_mask = 1.0 - clean_mask  # 0 on clean frames, 1 on noisy frames
        v_loss = (diff * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)
        v_loss = v_loss.mean()
        if not return_pred:
            return v_loss

        x_pred = z + (1.0 - t_view) * v_pred
        clean_mask_5d = clean_mask.view(b, 1, t, 1, 1)
        x_pred = x_pred * (1.0 - clean_mask_5d) + latents * clean_mask_5d
        return v_loss, x_pred.detach(), t_frame.max(dim=1).values.detach()


class DiffusionForcingDenoiser(Denoiser):
    def forward(
        self,
        latents: torch.Tensor,
        cond_seq: torch.Tensor,
        history_len: int = 1,
        return_pred: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one diffusion-forcing training step.

        Returns the scalar loss, or ``(loss, x_pred, t_noise)`` when
        ``return_pred`` is set: the detached ``(B, C, T, H, W)`` one-step clean
        latent and the ``(B,)`` peak noise level, for logging videos.
        """
        return super().forward_diffusion_forcing(
            latents=latents,
            cond_seq=cond_seq,
            history_len=history_len,
            return_pred=return_pred,
        )

    # ------------------------------------------------------------------
    #  Streaming AR-diffusion inference with KV cache
    # ------------------------------------------------------------------
    @staticmethod
    def _append_kv_cache(
        cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        new_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        depth = len(cache)
        out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * depth
        for i in range(depth):
            k_new, v_new = new_kv[i]
            if cache[i] is None:
                out[i] = (k_new, v_new)
            elif hasattr(cache[i], 'append'):
                out[i] = cache[i].append(k_new, v_new)
            else:
                k_old, v_old = cache[i]
                out[i] = (
                    torch.cat([k_old, k_new], dim=-2),
                    torch.cat([v_old, v_new], dim=-2),
                )
        return out

    @staticmethod
    def _evict_and_shift_cache(
        cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        drop_frames: int,
        tokens_per_frame: int,
        rope_module,
        sink_frames: int = 0,
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Evict ``drop_frames`` frames from the cache and renumber positions.

        With ``sink_frames == 0`` (default): drop the leading ``drop_frames``
        frames and re-rotate the remaining K so positions restart at 0 (pure
        FIFO sliding window).

        With ``sink_frames > 0`` (StreamingLLM-style attention sink): the first
        ``sink_frames`` frames are pinned at positions ``[0, sink_frames)`` and
        never dropped or re-rotated; only the *post-sink* window frames are
        evicted (oldest first) and shifted down by ``drop_frames`` so they sit
        contiguously right behind the sink (positions ``[sink_frames, ...)``).
        Net layout stays contiguous ``[0, cache_frames)`` and inside the trained
        RoPE range, while the true origin frame(s) stay resident as an anchor.
        """
        if drop_frames <= 0:
            return cache
        sink_frames = max(0, sink_frames)
        sink_tokens = sink_frames * tokens_per_frame
        drop_tokens = drop_frames * tokens_per_frame
        depth = len(cache)
        out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * depth
        for i in range(depth):
            if cache[i] is None:
                continue
            if hasattr(cache[i], 'evict'):
                out[i] = cache[i].evict(drop_frames, rope_module)
                continue
            k_old, v_old = cache[i]
            # sink slice: kept verbatim (no eviction, no RoPE shift).
            k_sink = k_old[..., :sink_tokens, :]
            v_sink = v_old[..., :sink_tokens, :]
            # window slice: drop the oldest ``drop_frames`` right after the sink,
            # then renumber survivors down by ``drop_frames``.
            k_rest = k_old[..., sink_tokens + drop_tokens:, :]
            v_rest = v_old[..., sink_tokens + drop_tokens:, :]
            if k_rest.numel() > 0:
                k_rest = rope_module.rope_shift_time(-drop_frames, k_rest)
            if sink_tokens > 0:
                k_new = torch.cat([k_sink, k_rest], dim=-2)
                v_new = torch.cat([v_sink, v_rest], dim=-2)
            else:
                k_new, v_new = k_rest, v_rest
            out[i] = (k_new, v_new)
        return out

    @torch.no_grad()
    def generate_eval_latents_streaming(
        self,
        latents: torch.Tensor,
        cond_seq: torch.Tensor,
        total_len: int,
        history_len: int = 1,
        max_cache_chunks: int = 16,
        inflight_chunks: int = 4,
        sink_frames: int = 0,
        stream_decoder=None,
        collect_stream_timing: bool = False,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Streaming AR-diffusion with KV cache (position-bounded, renumbered from 0).

        The active attention window at any time is exactly
        ``(max_cache_chunks + inflight_chunks) * df_chunk_size`` frames, which must
        fit inside ``self.trained_num_frames`` to avoid RoPE extrapolation.

        Committed chunks live in a per-block KV cache, logically numbered at
        temporal positions ``[0, cache_frames)``.  In-flight chunks sit at
        ``[cache_frames, cache_frames + inflight_frames)``.  When a chunk
        completes denoising (its ``t`` hits 0) it is committed: we run a t=0
        forward to obtain its K/V, append them to the cache, and if the cache
        overflows we drop the leading frames and :meth:`rope_shift_time` the
        remaining K to renumber positions back to 0.

        When ``stream_decoder`` is provided (Wan2.2 ``StreamingVAEDecoder``), each
        committed latent chunk is VAE-decoded immediately (decode-on-commit).
        Concatenating those RGB chunks is bit-exact with batch ``vae_decode`` of
        the same latents, so clip metrics are unchanged.

        Args:
            latents: ``(B, C, T_full, H, W)`` -- first ``history_len`` frames are
                used as clean visual context.
            cond_seq: ``(B, T_full, D)`` full action sequence.
            total_len: number of latent frames to produce.
            history_len: clean context length in frames; must be ``> 0`` and at
                most ``max_cache_chunks * df_chunk_size`` for full-chunk
                prefill. When ``history_len % df_chunk_size != 0`` (e.g.
                image-to-video with ``history_len=1`` and ``df_chunk_size=2``),
                the leading ``history_len // df_chunk_size`` chunks are
                pre-filled into the KV cache, and the remaining
                ``history_len % df_chunk_size`` frames are pinned to ``t=0``
                inside the first in-flight chunk.
            max_cache_chunks: max number of committed chunks retained in the
                cache at any one time.
            inflight_chunks: number of chunks simultaneously being denoised.
            sink_frames: StreamingLLM-style attention-sink size in frames. 0
                (default) = pure sliding window (no resident anchor). >0 pins the
                first ``sink_frames`` committed frames (the true origin / clean
                context) at cache positions ``[0, sink_frames)`` permanently;
                they are never evicted or re-rotated, so a long rollout always
                retains them as an anchor. Must be <= the cache capacity.
            stream_decoder: optional streaming VAE decoder with
                ``begin() / step(latents) / end()``. When set, returns
                ``(latents, rgb_video)`` with RGB in ``[-1, 1]``; otherwise
                returns latents only.
            noise: optional ``(B, C, >=total_len, H, W)`` initial noise; pass a
                fixed tensor to make repeated rollouts comparable. Sampled from
                the global RNG when omitted.
        """
        del kwargs
        device = latents.device
        dtype = latents.dtype
        net = self.net
        chunk_size = self.df_chunk_size
        inflight_frames = inflight_chunks * chunk_size
        max_cache_frames = max_cache_chunks * chunk_size
        active_frames = max_cache_frames + inflight_frames
        sink_frames = max(0, int(sink_frames))
        assert sink_frames <= max_cache_frames, (
            f"[StreamingGen] sink_frames={sink_frames} exceeds cache capacity "
            f"max_cache_frames={max_cache_frames}. Increase stream_max_cache_chunks."
        )

        trained_num_frames = int(getattr(self, "trained_num_frames", 0))
        if trained_num_frames <= 0:
            trained_num_frames = self.cfg.latent_frames
        assert active_frames <= trained_num_frames, (
            f"[StreamingGen] active window (cache={max_cache_frames} + "
            f"inflight={inflight_frames} = {active_frames}) exceeds "
            f"trained_num_frames={trained_num_frames}. Reduce "
            f"--stream_max_cache_chunks or --stream_inflight_chunks."
        )
        assert not net.use_abs_pos, (
            "[StreamingGen] requires use_abs_pos=False (relative RoPE only). "
            "MiniWorld checkpoints should be trained with RoPE-only positioning."
        )
        assert self.df_ardiff_step > 0, (
            "[StreamingGen] requires df_ardiff_step > 0 (AR-diffusion schedule)."
        )

        total_len = min(total_len, latents.shape[2], cond_seq.shape[1])

        ctx_len = int(history_len)
        assert ctx_len > 0, f"[StreamingGen] history_len must be > 0, got {ctx_len}"
        # Sub-chunk ctx (e.g. ctx_len=1, chunk_size=2 for i2v) is supported via
        # per-frame t=0 pinning inside the first in-flight chunk; we only
        # pre-fill the *whole* leading chunks into the KV cache. The leftover
        # ``ctx_len - n_full_ctx_frames`` frames stay in the in-flight window
        # cache capacity.
        n_full_ctx_chunks = ctx_len // chunk_size
        n_full_ctx_frames = n_full_ctx_chunks * chunk_size
        assert n_full_ctx_frames <= max_cache_frames, (
            f"[StreamingGen] full-chunk history ({n_full_ctx_frames} frames "
            f"= {n_full_ctx_chunks} chunks) exceeds cache capacity "
            f"({max_cache_frames} frames = {max_cache_chunks} chunks). "
            f"Increase --stream_max_cache_chunks."
        )

        b, c_ch, _, h, w = latents.shape
        p_t, p_h, p_w = net.x_embedder.patch_size
        _, h_total, w_total = net.x_embedder.input_size
        grid_h = h_total // p_h
        grid_w = w_total // p_w
        tokens_per_frame = grid_h * grid_w
        rope_module = net.feat_rope

        use_cfg = float(self.cfg_scale) > 1.0
        depth = net.depth
        kv_runtime = None
        if self.cfg.kv_acceleration == 'forcing-kv':
            if self.cfg.headcast:
                raise ValueError('Choose kv_acceleration or legacy headcast, not both')
            from miniworld.forcing_kv import ForcingKVConfig, ForcingKVRuntime
            signature = {name: getattr(self.cfg, name, None) for name in (
                'wm_model', 'cond_dim', 'cond_per_token', 'cond_dropout_prob',
                'action_null_first', 'adaln_mode', 'wm_use_qknorm', 'wm_mlp_ratio')}
            signature.update(cfg_scale=float(self.cfg_scale), num_sampling_steps=int(self.steps),
                             max_cache_frames=max_cache_frames, inflight_chunks=inflight_chunks,
                             trained_frames=trained_num_frames,
                             timestep_shift=float(self.timestep_shift),
                             df_ardiff_step=int(self.df_ardiff_step),
                             use_model_null_cfg=bool(self.use_model_null_cfg),
                             cfg_interval_min=float(self.cfg_interval_min),
                             cfg_interval_max=float(self.cfg_interval_max))
            kv_runtime = ForcingKVRuntime(
                ForcingKVConfig(calibration_frames=self.cfg.kv_calibration_frames,
                    threshold=self.cfg.forcing_kv_threshold,
                    query_samples_per_chunk=self.cfg.kv_query_samples_per_chunk,
                    sink_frames=sink_frames, segments=self.cfg.forcing_kv_segments,
                    retention=self.cfg.forcing_kv_retention,
                    profile_path=self.cfg.forcing_kv_profile,
                    profile_output=self.cfg.forcing_kv_profile_output,
                    model_identity=self.cfg.forcing_kv_model_identity,
                    profile_metadata=signature),
                depth=depth, num_heads=net.num_heads, height=grid_h, width=grid_w,
                chunk_size=chunk_size, max_cache_frames=max_cache_frames,
                branches=('cond', 'uncond') if use_cfg else ('cond',))
        elif self.cfg.kv_acceleration != 'off':
            if self.cfg.headcast:
                raise ValueError('Choose kv_acceleration or legacy headcast, not both')
            from miniworld.kv_acceleration import KVAccelerationConfig, KVAccelerationRuntime
            kv_runtime = KVAccelerationRuntime(
                KVAccelerationConfig(
                    mode=self.cfg.kv_acceleration,
                    calibration_frames=self.cfg.kv_calibration_frames,
                    max_relative_rmse=self.cfg.kv_max_relative_rmse,
                    min_cosine=self.cfg.kv_min_cosine,
                    spatial_score_threshold=self.cfg.kv_spatial_score_threshold,
                    query_samples_per_chunk=self.cfg.kv_query_samples_per_chunk,
                    sink_frames=sink_frames,
                    dummy_frames=self.cfg.kv_dummy_frames,
                    local_frames=self.cfg.kv_recent_frames,
                    tile_height=self.cfg.kv_tile_height,
                    tile_width=self.cfg.kv_tile_width,
                    dummy_fraction=self.cfg.kv_dummy_fraction,
                ), depth=depth, num_heads=net.blocks[0].attn.num_heads,
                height=grid_h, width=grid_w, chunk_size=chunk_size,
                max_cache_frames=max_cache_frames,
                branches=('cond', 'uncond') if use_cfg else ('cond',),
            )
        headcast = None
        if self.cfg.headcast:
            from miniworld.headcast import HeadCastConfig, HeadCastRuntime
            headcast = HeadCastRuntime(
                HeadCastConfig(
                    enabled=True,
                    tile_height=self.cfg.headcast_tile_height,
                    tile_width=self.cfg.headcast_tile_width,
                    recent_chunks=self.cfg.headcast_recent_chunks,
                    max_relative_rmse=self.cfg.headcast_max_relative_rmse,
                    min_cosine=self.cfg.headcast_min_cosine,
                    query_samples_per_chunk=self.cfg.headcast_query_samples_per_chunk,
                ),
                depth=depth, num_heads=net.blocks[0].attn.num_heads,
                grid_height=grid_h, grid_width=grid_w, chunk_size=chunk_size,
                sink_frames=sink_frames, max_cache_frames=max_cache_frames,
                branches=('cond', 'uncond') if use_cfg else ('cond',),
            )
        cache_cond: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * depth
        cache_uncond: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * depth if use_cfg else []
        cache_frames = 0

        # --- Output buffer ---
        if noise is None:
            z_global = torch.randn(b, c_ch, total_len, h, w, device=device, dtype=dtype)
        else:
            assert noise.shape[:2] == (b, c_ch) and noise.shape[3:] == (h, w), (
                f"[StreamingGen] noise shape {tuple(noise.shape)} does not match "
                f"latents {tuple(latents.shape)}"
            )
            assert noise.shape[2] >= total_len, (
                f"[StreamingGen] noise covers {noise.shape[2]} frames, need {total_len}"
            )
            # Cloned because the rollout denoises this buffer in place.
            z_global = noise[:, :, :total_len].to(device=device, dtype=dtype).clone()
        if ctx_len > 0:
            z_global[:, :, :ctx_len] = latents[:, :, :ctx_len]

        timing_enabled = bool(collect_stream_timing)
        timing_start = None
        dit_chunk_events: List[Dict[str, object]] = []
        vae_chunk_events: List[Dict[str, object]] = []

        def _timing_now() -> float:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            return time.perf_counter()

        if timing_enabled:
            timing_start = _timing_now()

        # --- Optional streaming VAE decode (decode-on-commit) ---
        rgb_parts: List[torch.Tensor] = []
        decoded_frames = 0

        def _stream_decode_upto(end_frame: int, *, chunk_idx: Optional[int] = None, step_idx: Optional[int] = None) -> None:
            nonlocal decoded_frames
            if stream_decoder is None or end_frame <= decoded_frames:
                return
            start_frame = decoded_frames
            t0 = _timing_now() if timing_enabled else None
            rgb_parts.append(stream_decoder.step(z_global[:, :, start_frame:end_frame]))
            t1 = _timing_now() if timing_enabled else None
            if timing_enabled and timing_start is not None and t0 is not None and t1 is not None:
                vae_chunk_events.append(
                    {
                        "chunk_idx": int(chunk_idx) if chunk_idx is not None else None,
                        "step_idx": int(step_idx) if step_idx is not None else None,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "generated": bool(chunk_idx is not None and chunk_idx >= n_full_ctx_chunks),
                        "start_sec": float(t0 - timing_start),
                        "end_sec": float(t1 - timing_start),
                        "duration_sec": float(t1 - t0),
                    }
                )
            decoded_frames = end_frame

        if stream_decoder is not None:
            stream_decoder.begin()

        try:
            # --- Pre-fill cache with clean history context ---
            # Only fully-aligned ctx chunks go into the cache. Sub-chunk leftover
            # (n_partial_ctx_frames) is pinned via per-frame t=0 inside the first
            # in-flight chunk, see the in-flight forward block below.
            if n_full_ctx_frames > 0:
                ctx_frames = z_global[:, :, :n_full_ctx_frames]
                ctx_cond = cond_seq[:, :n_full_ctx_frames]
                ctx_t = torch.zeros(b, n_full_ctx_frames, device=device, dtype=dtype)

                _, kv_cond_ctx = net.forward_with_cache(
                    ctx_frames, ctx_t, ctx_cond,
                    past_kv_list=None, current_position_offset=0,
                    return_kv=True, chunk_size=chunk_size,
                )
                cache_cond = list(kv_cond_ctx)
                if use_cfg:
                    ctx_uncond, ctx_drop_uncond = self._make_uncond(ctx_cond)
                    _, kv_uncond_ctx = net.forward_with_cache(
                        ctx_frames, ctx_t, ctx_uncond,
                        past_kv_list=None, current_position_offset=0,
                        return_kv=True, chunk_size=chunk_size, cond_drop=ctx_drop_uncond,
                    )
                    cache_uncond = list(kv_uncond_ctx)
                cache_frames = n_full_ctx_frames
                # Decode clean context immediately (same order as batch decode).
                _stream_decode_upto(n_full_ctx_frames)

            # --- Global AR schedule ---
            # Keep the final partial chunk. Training uses the same chunk layout
            # (e.g. T=9, chunk_size=2 -> four 2-frame chunks plus one 1-frame
            # chunk), so dropping it at inference changes the requested video
            # length and the learned schedule.
            total_chunks = (total_len + chunk_size - 1) // chunk_size
            # Residence cap: a chunk can be updated for ~inflight*ar outer steps
            # before the FIFO window must slide past it. When the *entire*
            # sequence fits in the inflight window, nothing is force-evicted
            # mid-denoise, so use the full sampler length (e.g. T=64, 100 steps).
            residence_cap = inflight_chunks * max(self.df_ardiff_step, 1)
            if total_chunks <= inflight_chunks:
                effective_steps = int(self.steps)
            else:
                effective_steps = min(int(self.steps), residence_cap)
            t_chunk_sched, t_next_chunk_sched, chunk_update_mask = (
                self._build_chunk_sampling_schedule(
                    total_chunks=total_chunks,
                    device=device, dtype=dtype,
                    n_context_chunks=n_full_ctx_chunks,
                    effective_steps=effective_steps,
                )
            )
            valid_intervals = self._compute_fifo_valid_intervals(
                chunk_update_mask, total_chunks, max_chunks_in_window=inflight_chunks,
            )

            num_outer_steps = t_chunk_sched.shape[0]
            self._set_last_eval_meta(
                path="streaming",
                total_chunks=total_chunks,
                n_ctx_chunks=n_full_ctx_chunks,
                num_outer_steps=num_outer_steps,
                effective_steps=effective_steps,
            )
            self.last_eval_meta["cfg_enabled"] = bool(use_cfg)
            self.last_eval_meta["stream_timing_enabled"] = bool(timing_enabled)
            _print0(f"[StreamingGen] total_len={total_len}, total_chunks={total_chunks}, "
                    f"ctx_chunks={n_full_ctx_chunks}, chunk_size={chunk_size}, "
                    f"inflight_chunks={inflight_chunks}, max_cache_chunks={max_cache_chunks}, "
                    f"trained_num_frames={trained_num_frames}, active_frames={active_frames}, "
                    f"sink_frames={sink_frames}, "
                    f"effective_steps={effective_steps}, outer_steps={num_outer_steps}, "
                    f"ar_step={self.df_ardiff_step}, "
                    f"stream_decode={stream_decoder is not None}, cfg_enabled={use_cfg}")

            last_win_sc = n_full_ctx_chunks
            committed_chunks = set(range(n_full_ctx_chunks))
            # VAE decode is tied to schedule completion (t_next==0), not KV
            # eviction. With max_cache=0 + full inflight, the window may never
            # slide, but chunks still finish and should decode immediately.
            next_decode_ci = n_full_ctx_chunks

            def _decode_finished_chunks(step_idx: int) -> None:
                nonlocal next_decode_ci
                if stream_decoder is None:
                    return
                while next_decode_ci < total_chunks:
                    if float(t_next_chunk_sched[step_idx, next_decode_ci]) > 0.0:
                        break
                    end_f = min((next_decode_ci + 1) * chunk_size, total_len)
                    if timing_enabled and timing_start is not None:
                        t_dit = _timing_now()
                        dit_chunk_events.append(
                            {
                                "chunk_idx": int(next_decode_ci),
                                "step_idx": int(step_idx),
                                "end_frame": int(end_f),
                                "generated": bool(next_decode_ci >= n_full_ctx_chunks),
                                "complete_sec": float(t_dit - timing_start),
                            }
                        )
                    _stream_decode_upto(end_f, chunk_idx=next_decode_ci, step_idx=step_idx)
                    _print0(
                        f"[StreamingGen]   decoded chunk {next_decode_ci}/{total_chunks} | "
                        f"latent_frames={decoded_frames}/{total_len} | "
                        f"step={step_idx}/{num_outer_steps}"
                    )
                    next_decode_ci += 1

            for step in range(num_outer_steps):
                win_sc, win_ec = valid_intervals[step]
                # Enforce: committed chunks stay inside cache coverage.
                # win_sc should equal committed chunks count.  If win_sc < committed
                # (shouldn't happen), clamp.
                win_sc = max(win_sc, n_full_ctx_chunks)

                # --- Commit newly-finished chunks into the cache ---
                while last_win_sc < win_sc:
                    ci = last_win_sc
                    gsl = slice(ci * chunk_size, min((ci + 1) * chunk_size, total_len))
                    commit_frames = z_global[:, :, gsl]
                    commit_cond = cond_seq[:, gsl]
                    # t=0: chunk has finished denoising, treat as clean ctx going forward.
                    t_commit = torch.zeros(
                        b, commit_frames.shape[2], device=device, dtype=dtype,
                    )

                    _, kv_cond_new = net.forward_with_cache(
                        commit_frames, t_commit, commit_cond,
                        past_kv_list=cache_cond,
                        current_position_offset=cache_frames,
                        return_kv=True, chunk_size=chunk_size,
                        **({} if kv_runtime is None else {'kv_context': kv_runtime.begin_forward(
                            'cond', cache_frames, t_commit, commit=True)}),
                    )
                    cache_cond = self._append_kv_cache(cache_cond, kv_cond_new)
                    if use_cfg:
                        commit_uncond, commit_drop_uncond = self._make_uncond(commit_cond)
                        _, kv_uncond_new = net.forward_with_cache(
                            commit_frames, t_commit, commit_uncond,
                            past_kv_list=cache_uncond,
                            current_position_offset=cache_frames,
                            return_kv=True, chunk_size=chunk_size, cond_drop=commit_drop_uncond,
                            **({} if kv_runtime is None else {'kv_context': kv_runtime.begin_forward(
                                'uncond', cache_frames, t_commit, commit=True)}),
                        )
                        cache_uncond = self._append_kv_cache(cache_uncond, kv_uncond_new)
                    cache_frames += commit_frames.shape[2]

                    if cache_frames > max_cache_frames:
                        # Never evict into the resident sink region.
                        drop = min(cache_frames - max_cache_frames,
                                   cache_frames - sink_frames)
                        if drop > 0:
                            cache_cond = self._evict_and_shift_cache(
                                cache_cond, drop, tokens_per_frame, rope_module,
                                sink_frames=sink_frames,
                            )
                            if use_cfg:
                                cache_uncond = self._evict_and_shift_cache(
                                    cache_uncond, drop, tokens_per_frame, rope_module,
                                    sink_frames=sink_frames,
                                )
                            cache_frames -= drop

                    if kv_runtime is not None:
                        cache_cond = kv_runtime.compact(cache_cond, cache_frames, branch='cond')
                        if use_cfg:
                            cache_uncond = kv_runtime.compact(cache_uncond, cache_frames, branch='uncond')

                    committed_chunks.add(ci)
                    _print0(f"[StreamingGen]   committed chunk {ci}/{total_chunks} | "
                            f"cache_frames={cache_frames} | step={step}/{num_outer_steps}")
                    last_win_sc += 1

                if win_ec <= win_sc:
                    _decode_finished_chunks(step)
                    continue

                # --- In-flight forward ---
                win_sf = win_sc * chunk_size
                win_ef = min(win_ec * chunk_size, total_len)
                inflight_z = z_global[:, :, win_sf:win_ef].clone()
                inflight_cond = cond_seq[:, win_sf:win_ef]

                n_inflight = win_ec - win_sc
                inflight_chunk_slices = self._build_chunk_slices(win_ef - win_sf)

                t_chunks = t_chunk_sched[step, win_sc:win_ec]
                t_next_chunks = t_next_chunk_sched[step, win_sc:win_ec]
                t_frame = self._broadcast_chunk_values_to_frames(
                    t_chunks.unsqueeze(0).expand(b, -1),
                    inflight_chunk_slices, win_ef - win_sf,
                )
                t_next_frame = self._broadcast_chunk_values_to_frames(
                    t_next_chunks.unsqueeze(0).expand(b, -1),
                    inflight_chunk_slices, win_ef - win_sf,
                )

                # Pin sub-chunk context frames inside this window to t=0 so dt=0
                # and they aren't perturbed by the velocity update.
                ctx_in_inflight = min(max(0, ctx_len - win_sf), win_ef - win_sf)
                if ctx_in_inflight > 0:
                    t_frame[:, :ctx_in_inflight] = 0.0
                    t_next_frame[:, :ctx_in_inflight] = 0.0
                    inflight_z[:, :, :ctx_in_inflight] = latents[
                        :, :, win_sf:win_sf + ctx_in_inflight
                    ]

                headcast_cond = None if headcast is None else headcast.begin_forward('cond', cache_frames, t_frame)
                headcast_cond_kwargs = {} if headcast_cond is None else {'headcast_context': headcast_cond}
                kv_cond = None if kv_runtime is None else kv_runtime.begin_forward('cond', cache_frames, t_frame)
                if kv_cond is not None:
                    headcast_cond_kwargs['kv_context'] = kv_cond
                v_cond_pred, _ = net.forward_with_cache(
                    inflight_z, t_frame, inflight_cond,
                    past_kv_list=cache_cond,
                    current_position_offset=cache_frames,
                    return_kv=False, chunk_size=chunk_size,
                    **headcast_cond_kwargs,
                )
                if headcast is not None:
                    headcast.finish_forward(headcast_cond)
                if kv_runtime is not None:
                    kv_runtime.finish_forward(kv_cond)
                if use_cfg:
                    inflight_uncond, inflight_drop_uncond = self._make_uncond(inflight_cond)
                    headcast_uncond = headcast.begin_forward('uncond', cache_frames, t_frame) if headcast is not None else None
                    headcast_uncond_kwargs = {} if headcast_uncond is None else {'headcast_context': headcast_uncond}
                    kv_uncond = None if kv_runtime is None else kv_runtime.begin_forward('uncond', cache_frames, t_frame)
                    if kv_uncond is not None:
                        headcast_uncond_kwargs['kv_context'] = kv_uncond
                    v_uncond_pred, _ = net.forward_with_cache(
                        inflight_z, t_frame, inflight_uncond,
                        past_kv_list=cache_uncond,
                        current_position_offset=cache_frames,
                        return_kv=False, chunk_size=chunk_size, cond_drop=inflight_drop_uncond,
                        **headcast_uncond_kwargs,
                    )
                    if headcast is not None:
                        headcast.finish_forward(headcast_uncond)
                    if kv_runtime is not None:
                        kv_runtime.finish_forward(kv_uncond)
                else:
                    v_uncond_pred = None

                if kv_runtime is not None:
                    cache_cond = kv_runtime.compact(cache_cond, cache_frames, branch='cond')
                    if use_cfg:
                        cache_uncond = kv_runtime.compact(cache_uncond, cache_frames, branch='uncond')

                update_mask_row = chunk_update_mask[step, win_sc:win_ec]
                for lci in range(n_inflight):
                    if not bool(update_mask_row[lci]):
                        continue
                    sl = inflight_chunk_slices[lci]
                    gsl = slice(
                        (win_sc + lci) * chunk_size,
                        min((win_sc + lci + 1) * chunk_size, total_len),
                    )
                    gl = gsl.stop - gsl.start
                    if use_cfg:
                        chunk_t_val = t_frame[:, sl].mean(dim=1)
                        action_scale = self._get_df_action_guidance_scale(chunk_t_val)
                        action_scale = action_scale.view(-1, 1, 1, 1, 1)
                        assert v_uncond_pred is not None
                        v_chunk = (
                            v_uncond_pred[:, :, sl]
                            + action_scale * (v_cond_pred[:, :, sl] - v_uncond_pred[:, :, sl])
                        )
                    else:
                        v_chunk = v_cond_pred[:, :, sl]
                    dt = (t_next_frame[:, sl] - t_frame[:, sl]).view(b, 1, -1, 1, 1)[:, :, :gl]
                    z_global[:, :, gsl] = (
                        z_global[:, :, gsl] - dt * v_chunk[:, :, :gl]
                    )

                # Re-pin clean ctx frames (numerical safety; dt should already be
                # 0 for them, but FP error can drift otherwise).
                if ctx_len > 0:
                    z_global[:, :, :ctx_len] = latents[:, :, :ctx_len]

                # Decode as soon as each leading chunk's schedule hits t=0.
                _decode_finished_chunks(step)

            # Flush any remaining (e.g. final partial) frames for VAE.
            if decoded_frames < total_len:
                flush_ci = (total_len - 1) // chunk_size
                if timing_enabled and timing_start is not None:
                    t_dit = _timing_now()
                    dit_chunk_events.append(
                        {
                            "chunk_idx": int(flush_ci),
                            "step_idx": int(num_outer_steps),
                            "end_frame": int(total_len),
                            "generated": bool(flush_ci >= n_full_ctx_chunks),
                            "complete_sec": float(t_dit - timing_start),
                        }
                    )
                _stream_decode_upto(total_len, chunk_idx=flush_ci, step_idx=num_outer_steps)
            if timing_enabled and timing_start is not None:
                self.last_eval_meta["stream_timing"] = {
                    "enabled": True,
                    "cfg_enabled": bool(use_cfg),
                    "start_sec": 0.0,
                    "dit_chunk_events": dit_chunk_events,
                    "vae_chunk_events": vae_chunk_events,
                }
            _print0(
                f"[StreamingGen] done. committed={len(committed_chunks)}/{total_chunks} "
                f"chunks, decoded_latent_frames={decoded_frames}/{total_len}, "
                f"effective_steps={effective_steps}."
            )

            if stream_decoder is not None:
                assert rgb_parts, (
                    "[StreamingGen] stream_decoder was set but no RGB chunks were produced"
                )
                return z_global, torch.cat(rgb_parts, dim=2)
            return z_global
        finally:
            if kv_runtime is not None:
                kv_runtime.record_storage(cache_cond, cache_uncond)
                self.last_eval_meta['kv_acceleration'] = kv_runtime.summary()
            if headcast is not None:
                self.last_eval_meta['headcast'] = headcast.summary()
            if stream_decoder is not None:
                stream_decoder.end()


def build_denoiser_from_mode(cfg: DenoiserConfig) -> Denoiser:
    """Build the public MiniWorld AR-diffusion denoiser."""
    return DiffusionForcingDenoiser(cfg)
