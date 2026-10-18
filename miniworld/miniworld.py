"""MiniWorld: action / pose-conditioned streaming video DiT.

The model combines RoPE-only video tokens, action/pose conditioning streams,
AdaLN-LoRA modulation, and structured condition dropout for streaming world
modeling.

  * **RoPE-only** positioning (absolute ``pos_embed`` removed entirely) so
    train / streaming inference share the same position scheme.
  * **AdaLN-LoRA modulation** (``adaln_mode="adaln_lora"``, default): a single
    model-level modulation MLP produces a shared ``(B, T, 6D)`` term reused by
    every block, plus a cheap per-block low-rank refinement ``D -> r -> 6D``.
    This is the parameter-efficient middle ground between FLUX.2's
    fully-shared modulation and the classic per-block full ``D -> 6D`` AdaLN.
    Two more modes are provided for ablation: ``"fully_shared"`` (FLUX.2 style)
    and ``"per_block"`` (classic DiT).
  * **Separated conditioning streams** instead of the old
    ``c_token = t_emb + y_emb``:
      - timestep  -> its own encoder, drives the base modulation;
      - action    -> DreamDojo-style encoder, *added* into the timestep /
        AdaLN stream at per-latent-frame granularity (global-per-frame signal);
      - pose      -> ray-encoding, injected as a *separate* per-token spatial
        modulation stream (lingbot-style), kept out of the timestep stream.
  * **Structured condition dropout** with a learned null embedding, for
    classifier-free guidance training.

The forward signature is designed for ``miniworld.denoiser.Denoiser``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from flash_attn import flash_attn_func

from miniworld.optimization_config import OptimizationConfig, load_optimization_config

# FlexAttention is optional; disabled by default
create_block_mask = None
flex_attention = None


# --------------------------------------------------------------------------- #
#                           Attention masks (streaming)                        #
# --------------------------------------------------------------------------- #
def _build_temporal_chunkwise_attn_mask(
    seq_len: int,
    tokens_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> torch.Tensor:
    """Block-causal (chunk-wise) additive mask over the temporal axis."""
    token_idx = torch.arange(seq_len, device=device)
    frame_idx = token_idx // tokens_per_frame
    chunk_idx = frame_idx // chunk_size
    mask = chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)
    float_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    float_mask.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    return float_mask


def _build_cached_block_causal_mask(
    n_past: int,
    n_cur: int,
    tokens_per_frame: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask for streaming forward with a KV cache.

    Query length is ``n_cur`` (in-flight tokens); key/value length is
    ``n_past + n_cur``.  Past cache is always visible; current tokens use
    block-causal attention along the temporal chunk axis.
    """
    total_kv = n_past + n_cur
    float_mask = torch.zeros((1, 1, n_cur, total_kv), device=device, dtype=dtype)
    if n_cur == 0:
        return float_mask
    token_idx = torch.arange(n_cur, device=device)
    chunk_idx = (token_idx // tokens_per_frame) // chunk_size
    cur_mask = chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)  # (n_cur, n_cur)
    float_mask[0, 0, :, n_past:] = torch.where(
        cur_mask,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), float("-inf"), device=device, dtype=dtype),
    )
    return float_mask


def _chunkwise_flash_attention(q: Tensor, k: Tensor, v: Tensor, chunk_tokens: int) -> Tensor:
    """Cached block-causal attention on tensors shaped ``(B, heads, N, D)``.

    Each query chunk sees all cached tokens and the current-token prefix ending
    at that chunk. These are exactly the finite entries of the streaming mask;
    slicing them away preserves a single softmax over every visible key.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    n_cur = q.shape[-2]
    n_past = k.shape[-2] - n_cur
    if n_cur == 0 or n_past < 0 or k.shape != v.shape:
        raise ValueError("expected nonempty queries and matching K/V containing history + current tokens")

    # SDPA autocast computes in the active low-precision dtype even when Q/K
    # became FP32 in QK norm / RoPE. Cast only the attention inputs, once per
    # layer: returned/stored KV retains its original dtype and RoPE positions.
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else v.dtype
    q = q.transpose(1, 2).to(dtype=dtype)
    k = k.transpose(1, 2).to(dtype=dtype)
    v = v.transpose(1, 2).to(dtype=dtype)
    outputs = []
    for start in range(0, n_cur, chunk_tokens):
        end = min(start + chunk_tokens, n_cur)
        # The whole current chunk is bidirectional, including its later tokens.
        # causal=True would incorrectly introduce token-wise causality here.
        outputs.append(flash_attn_func(
            q[:, start:end], k[:, :n_past + end], v[:, :n_past + end], causal=False,
        ))
    # Cache commits commonly contain just one chunk; avoid a redundant copy.
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
    return output.transpose(1, 2)


# --------------------------------------------------------------------------- #
#                                 Basic layers                                 #
# --------------------------------------------------------------------------- #
def modulate(
    x: Tensor, shift: Optional[Tensor], scale: Tensor,
    conditioning_tokens_per_frame: Optional[int] = None,
) -> Tensor:
    """Apply AdaLN, optionally broadcasting compact (B, T, 1, D) modulation."""
    if conditioning_tokens_per_frame is not None:
        # Tokens are ordered by (frame, height, width). Splitting the token
        # axis is a view, including the channel-major patch-embedding layout.
        # Do not expand/reshape the modulation itself: that would materialize
        # the same per-frame vector once for every spatial token.
        framed = x.unflatten(1, (-1, conditioning_tokens_per_frame))
        return modulate(framed, shift, scale).flatten(1, 2)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
        if shift is not None:
            shift = shift.unsqueeze(1)
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def _apply_gate(
    update: Tensor, gate: Tensor, conditioning_tokens_per_frame: Optional[int],
) -> Tensor:
    """Keep frame gates compact while preserving the original multiply/add order."""
    if conditioning_tokens_per_frame is None:
        return gate * update
    return (gate * update.unflatten(1, (-1, conditioning_tokens_per_frame))).flatten(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class PatchEmbed3D(nn.Module):
    """(B, C, T, H, W) -> (B, N, D) via a 3D conv patchifier."""

    def __init__(
        self,
        input_size: int | Tuple[int, int, int],
        patch_size: int | Tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif len(patch_size) == 2:
            patch_size = (1, patch_size[0], patch_size[1])

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.num_patches = (
            (input_size[0] // patch_size[0])
            * (input_size[1] // patch_size[1])
            * (input_size[2] // patch_size[2])
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        rope: Optional[Callable] = None,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        return_kv: bool = False,
        cached_chunk_tokens: Optional[int] = None,
    ):
        B, N, C = x.shape
        in_dtype = x.dtype
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        k_current, v_current = k, v
        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past.to(dtype=k.dtype, device=k.device), k], dim=-2)
            v = torch.cat([v_past.to(dtype=v.dtype, device=v.device), v], dim=-2)

        if cached_chunk_tokens is not None:
            if attn_mask is not None:
                raise ValueError("cached_chunk_tokens cannot be combined with an arbitrary attention mask")
            # Projection, QK norm, RoPE and KV concatenation remain shared by
            # both backends; only the block-causal attention evaluation changes.
            x = _chunkwise_flash_attention(q, k, v, cached_chunk_tokens)
        elif attn_mask is None and past_kv is None:
            # flash-attn fast path expects (B, N, num_heads, head_dim)
            q = q.transpose(1, 2).to(torch.bfloat16)
            k = k.transpose(1, 2).to(torch.bfloat16)
            v = v.transpose(1, 2).to(torch.bfloat16)
            x = flash_attn_func(q, k, v, causal=False).transpose(1, 2)
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, C).to(in_dtype)
        x = self.proj_drop(self.proj(x))
        if return_kv:
            return x, (k_current, v_current)
        return x


# --------------------------------------------------------------------------- #
#                             3D rotary embedding                              #
# --------------------------------------------------------------------------- #
def broadcat(tensors, dim: int = -1):
    num_tensors = len(tensors)
    shape_lens = {len(t.shape) for t in tensors}
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims), "invalid broadcast dims"
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast3D(nn.Module):
    """Axial 3D RoPE (time / height / width), borrowed from EVA/lightning_wm."""

    def __init__(self, dim: int, num_frames: int, frame_height: int, frame_width: int, theta: int = 10000) -> None:
        super().__init__()
        dim_h = (dim // 3) // 2 * 2
        dim_w = (dim // 3) // 2 * 2
        dim_t = dim - dim_h - dim_w
        if dim_t % 2 != 0:
            dim_t -= 1

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        self.register_buffer("base_freqs_t", freqs_t)
        self.register_buffer("base_freqs_h", freqs_h)
        self.register_buffer("base_freqs_w", freqs_w)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.dim = dim
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        self._num_frames = num_frames

        freqs_cos, freqs_sin = self._build_freqs(
            num_frames, freqs_t, freqs_h, freqs_w, frame_height, frame_width, dim
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    @staticmethod
    def _build_freqs(num_frames, freqs_t, freqs_h, freqs_w, fh, fw, dim, start_frame: int = 0):
        device = freqs_t.device
        t_time = torch.arange(start_frame, start_frame + num_frames, device=device, dtype=torch.float32)
        t_height = torch.arange(fh, device=device, dtype=torch.float32)
        t_width = torch.arange(fw, device=device, dtype=torch.float32)
        ft = repeat(torch.einsum("n,d->nd", t_time, freqs_t), "... n -> ... (n r)", r=2)
        fht = repeat(torch.einsum("n,d->nd", t_height, freqs_h), "... n -> ... (n r)", r=2)
        fwt = repeat(torch.einsum("n,d->nd", t_width, freqs_w), "... n -> ... (n r)", r=2)
        freqs = broadcat(
            (ft.view(num_frames, 1, 1, -1), fht.view(1, fh, 1, -1), fwt.view(1, 1, fw, -1)), dim=-1
        )
        return freqs.cos().view(-1, dim), freqs.sin().view(-1, dim)

    def _get_freqs(self, num_frames: int, device: torch.device, start_frame: int = 0):
        if start_frame == 0 and num_frames == self._num_frames:
            return self.freqs_cos, self.freqs_sin
        return self._build_freqs(
            num_frames,
            self.base_freqs_t.to(device),
            self.base_freqs_h.to(device),
            self.base_freqs_w.to(device),
            self.frame_height,
            self.frame_width,
            self.dim,
            start_frame=start_frame,
        )

    def forward(self, t: Tensor, num_frames_override: Optional[int] = None, start_frame: int = 0) -> Tensor:
        num_frames = num_frames_override if num_frames_override is not None else self._num_frames
        cos, sin = self._get_freqs(num_frames, t.device, start_frame=start_frame)
        return t * cos + rotate_half(t) * sin

    def rope_shift_time(self, delta: int, cached: Tensor) -> Tensor:
        """Re-rotate cached K/Q on the temporal axis by ``delta`` frames.

        Given a tensor originally RoPE-rotated at temporal positions
        ``[p, ..., p+N-1]``, returns it rotated as if positions were
        ``[p+delta, ..., p+delta+N-1]``. Use ``delta=-k`` after evicting ``k``
        leading frames from a streaming cache to renumber positions back to 0.
        Only the temporal slice (first ``dim_t`` dims) is rotated; spatial dims
        receive identity rotation.
        """
        if delta == 0 or cached.numel() == 0:
            return cached
        device = cached.device
        out_dtype = cached.dtype

        base_freqs_t = self.base_freqs_t.to(device=device, dtype=torch.float32)
        angle_t = float(delta) * base_freqs_t
        angle_t_rep = repeat(angle_t, "n -> (n r)", r=2)
        cos_t = angle_t_rep.cos()
        sin_t = angle_t_rep.sin()

        rest = self.dim - self.dim_t
        cos_rest = torch.ones(rest, device=device, dtype=torch.float32)
        sin_rest = torch.zeros(rest, device=device, dtype=torch.float32)

        cos_full = torch.cat([cos_t, cos_rest], dim=-1).to(out_dtype)
        sin_full = torch.cat([sin_t, sin_rest], dim=-1).to(out_dtype)

        return cached * cos_full + rotate_half(cached) * sin_full


# --------------------------------------------------------------------------- #
#                              Timestep embedding                              #
# --------------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    """Scalar timestep -> D-dim vector (sinusoidal + MLP)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            return self.mlp(self.timestep_embedding(t, self.freq_dim))
        if t.dim() == 2:
            b, s = t.shape
            emb = self.mlp(self.timestep_embedding(t.reshape(-1), self.freq_dim))
            return emb.view(b, s, -1)
        raise ValueError(f"Unsupported timestep shape: {t.shape}")


# --------------------------------------------------------------------------- #
#                          Conditioning encoders                              #
# --------------------------------------------------------------------------- #
class ActionEncoder(nn.Module):
    """DreamDojo-style action encoder for a global per-frame action signal.

    Input action condition is ``(B, T, cond_dim)`` where each latent frame ``t``
    already packs its chunk of raw actions (the training pipeline builds
    ``cond_dim = num_action_per_latent * action_dim``).  Two MLP heads produce:

      * ``emb_B_T_D``  -- added into the timestep embedding stream;
      * ``mod_B_T_MD`` -- added into the (shared) AdaLN modulation stream,
        where ``M = n_mod_chunks`` (6 here: attn shift/scale/gate + mlp
        shift/scale/gate).

    This mirrors Cosmos' ``action_embedder_B_D`` / ``action_embedder_B_3D``
    but targets a 6-chunk modulation layout.
    """

    def __init__(self, cond_dim: int, hidden_size: int, n_mod_chunks: int = 6, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = hidden_size * hidden_mult
        act = lambda: nn.GELU(approximate="tanh")
        self.to_emb = nn.Sequential(
            nn.Linear(cond_dim, hidden), act(), nn.Linear(hidden, hidden_size)
        )
        self.to_mod = nn.Sequential(
            nn.Linear(cond_dim, hidden), act(), nn.Linear(hidden, n_mod_chunks * hidden_size)
        )

    def forward(self, action_B_T_C: Tensor) -> Tuple[Tensor, Tensor]:
        return self.to_emb(action_B_T_C), self.to_mod(action_B_T_C)


class PoseEncoder(nn.Module):
    """Ray-encoding -> per-token spatial AdaLN modulation (lingbot-style).

    The pose condition is a per-pixel ray-encoding volume
    ``(B, T, cond_dim, H_lat, W_lat)`` (see ``pose_utils.compute_ray_encoding``,
    e.g. cond_dim = 180 for origin+direction with 15 NeRF frequencies).

    Unlike ``ActionEncoder`` (a per-frame signal folded into the timestep
    stream), pose is spatially varying, so it drives its *own* per-token
    modulation stream ``(B, N, MD)`` that is added on top of the timestep /
    action modulation inside every block.  A residual MLP over the patchified
    ray features mirrors lingbot's ``cam_injector`` before producing scale/shift.
    """

    def __init__(self, cond_dim: int, hidden_size: int, patch_size: int, n_mod_chunks: int = 6) -> None:
        super().__init__()
        self.patchify = nn.Conv2d(cond_dim, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.res_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.to_mod = nn.Linear(hidden_size, n_mod_chunks * hidden_size, bias=True)

    def forward(self, pose_B_T_C_H_W: Tensor, b: int, grid_t: int) -> Tensor:
        """Return per-token modulation ``(B, N, MD)`` with N = grid_t * h * w."""
        y = rearrange(pose_B_T_C_H_W, "b t c h w -> (b t) c h w")
        y = self.patchify(y)  # ((B*T), D, h', w')
        y = rearrange(y, "(b t) d h w -> b (t h w) d", b=b, t=grid_t)
        y = y + self.res_mlp(y)  # residual (lingbot cam_injector style)
        return self.to_mod(y)  # (B, N, MD)


# --------------------------------------------------------------------------- #
#                          Modulation (shared / lora)                         #
# --------------------------------------------------------------------------- #
_MODES = ("adaln_lora", "fully_shared", "per_block")


class BlockModulation(nn.Module):
    """Per-block modulation producer, respecting the model-wide ``adaln_mode``.

    * ``adaln_lora``   : ``shared_mod + lora(emb)`` where ``lora = SiLU -> D->r
      -> r->MD`` (zero-init, so a block starts exactly at ``shared_mod``).
    * ``fully_shared`` : ``shared_mod`` (no per-block params; FLUX.2 style).
    * ``per_block``    : ``full(emb)`` with ``full = SiLU -> D->MD`` (classic
      per-block AdaLN, zero-init).
    """

    def __init__(self, hidden_size: int, adaln_mode: str, n_mod_chunks: int = 6, lora_dim: int = 256) -> None:
        super().__init__()
        assert adaln_mode in _MODES, f"adaln_mode must be one of {_MODES}"
        self.adaln_mode = adaln_mode
        out = n_mod_chunks * hidden_size
        if adaln_mode == "adaln_lora":
            self.lora = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, lora_dim, bias=False),
                nn.Linear(lora_dim, out, bias=False),
            )
        elif adaln_mode == "per_block":
            self.full = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, out, bias=True))
        # fully_shared: no parameters

    def forward(
        self, emb: Tensor, shared_mod: Optional[Tensor], pose_mod: Optional[Tensor],
        conditioning_tokens_per_frame: Optional[int] = None,
    ) -> Tensor:
        if self.adaln_mode == "adaln_lora":
            mod = shared_mod + self.lora(emb)
        elif self.adaln_mode == "fully_shared":
            mod = shared_mod
        else:  # per_block
            mod = self.full(emb)
        if conditioning_tokens_per_frame is not None:
            # Project only B*T rows. Keep the singleton spatial axis until an
            # elementwise operation consumes it; no dense modulation gather.
            mod = mod.unsqueeze(2)
            if pose_mod is not None:
                # In per_block/action mode this channel carries a frame delta;
                # actual ray-encoded pose deltas remain spatially varying.
                if pose_mod.shape[1] == emb.shape[1]:
                    pose_mod = pose_mod.unsqueeze(2)
                else:
                    pose_mod = pose_mod.unflatten(1, (-1, conditioning_tokens_per_frame))
        if pose_mod is not None:
            mod = mod + pose_mod
        return mod


# --------------------------------------------------------------------------- #
#                                   Blocks                                     #
# --------------------------------------------------------------------------- #
class MiniWorldBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        adaln_mode: str,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden))
        self.modulation = BlockModulation(hidden_size, adaln_mode, n_mod_chunks=6, lora_dim=lora_dim)

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        shared_mod: Optional[Tensor],
        pose_mod: Optional[Tensor],
        feat_rope: Optional[Callable] = None,
        attn_mask: Optional[Tensor] = None,
        past_kv: Optional[Tuple[Tensor, Tensor]] = None,
        return_kv: bool = False,
        cached_chunk_tokens: Optional[int] = None,
        conditioning_tokens_per_frame: Optional[int] = None,
    ):
        mod = self.modulation(emb, shared_mod, pose_mod, conditioning_tokens_per_frame)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        attn_result = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa, conditioning_tokens_per_frame),
            rope=feat_rope,
            attn_mask=attn_mask,
            past_kv=past_kv,
            return_kv=return_kv,
            cached_chunk_tokens=cached_chunk_tokens,
        )
        if return_kv:
            attn_out, new_kv = attn_result
        else:
            attn_out, new_kv = attn_result, None
        x = x + _apply_gate(attn_out, gate_msa, conditioning_tokens_per_frame)
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, conditioning_tokens_per_frame))
        x = x + _apply_gate(mlp_out, gate_mlp, conditioning_tokens_per_frame)
        if return_kv:
            return x, new_kv
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(
        self, x: Tensor, emb: Tensor, conditioning_tokens_per_frame: Optional[int] = None,
    ) -> Tensor:
        cond = self.adaLN_modulation(emb)
        if conditioning_tokens_per_frame is not None:
            cond = cond.unsqueeze(2)
        if cond.dim() == 2:
            shift, scale = cond.chunk(2, dim=1)
        else:
            shift, scale = cond.chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale, conditioning_tokens_per_frame))


# --------------------------------------------------------------------------- #
#                                Main model                                    #
# --------------------------------------------------------------------------- #
class MiniWorldModel(nn.Module):
    """Action / pose-conditioned streaming video DiT (RoPE-only)."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        cond_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        input_size: int | Tuple[int, int],
        num_frames: int = 9,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_checkpoint: bool = False,
        cond_per_token: bool = False,
        adaln_mode: str = "adaln_lora",
        adaln_lora_dim: int = 256,
        cond_dropout_prob: float = 0.0,
        action_null_first: bool = True,
        # Kept for checkpoint compatibility; MiniWorld always uses RoPE.
        use_rope: bool = True,
        use_abs_pos: bool = False,
        optimization_config: OptimizationConfig | str | Path | None = None,
    ) -> None:
        super().__init__()
        assert adaln_mode in _MODES, f"adaln_mode must be one of {_MODES}"
        assert use_rope, "MiniWorldModel is RoPE-only; use_rope must be True."
        assert not use_abs_pos, "MiniWorldModel is RoPE-only; use_abs_pos must be False."

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        # Per-model, immutable execution options. Loading here also covers
        # direct Python/profile callers; nothing is read from disk per forward
        # or registered in state_dict, so checkpoint keys remain unchanged.
        self.optimization_config = load_optimization_config(optimization_config)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cond_per_token = cond_per_token
        self.adaln_mode = adaln_mode
        self.cond_dropout_prob = cond_dropout_prob
        # Route the true first latent frame (the seed / initial observation,
        # which has no preceding action) through the learned ``null_action``.
        self.action_null_first = action_null_first
        # RoPE-only: kept as attributes for downstream code / streaming asserts.
        self.use_rope = True
        self.use_abs_pos = False

        input_size = (
            (num_frames, input_size, input_size)
            if isinstance(input_size, int)
            else (num_frames,) + tuple(input_size)
        )
        self.x_embedder = PatchEmbed3D(
            input_size=input_size,
            patch_size=(1, patch_size, patch_size) if isinstance(patch_size, int) else patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        # ---- conditioning streams -------------------------------------- #
        self.t_embedder = TimestepEmbedder(hidden_size)
        # RMSNorm on the (timestep [+ action]) embedding before it drives the
        # AdaLN heads / final layer. Mirrors Cosmos/DreamDojo ``t_embedding_norm``:
        # keeps the affine embedding well-conditioned once the action encoder
        # sums a second, independently-scaled signal into the timestep stream.
        self.emb_norm = RMSNorm(hidden_size)
        # shared modulation head (base AdaLN term reused across all blocks)
        self.shared_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        if cond_per_token:
            # pose (ray-encoding) -> separate per-token spatial modulation.
            self.pose_encoder = PoseEncoder(cond_dim, hidden_size, patch_size, n_mod_chunks=6)
            self.action_encoder = None
        else:
            # action -> DreamDojo-style, folded into timestep / AdaLN stream.
            self.action_encoder = ActionEncoder(cond_dim, hidden_size, n_mod_chunks=6)
            self.pose_encoder = None
            # learned null action for classifier-free guidance dropout.
            self.null_action = nn.Parameter(torch.zeros(1, 1, cond_dim))

        head_dim = hidden_size // num_heads
        self.feat_rope = VisionRotaryEmbeddingFast3D(
            dim=head_dim,
            num_frames=num_frames,
            frame_height=input_size[1] // patch_size,
            frame_width=input_size[2] // patch_size,
        )

        self.blocks = nn.ModuleList(
            [
                MiniWorldBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    adaln_mode=adaln_mode,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    lora_dim=adaln_lora_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    # ------------------------------------------------------------------ #
    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # AdaLN-zero: shared modulation starts at 0 -> identity blocks.
        nn.init.constant_(self.shared_mod[-1].weight, 0)
        nn.init.constant_(self.shared_mod[-1].bias, 0)

        # zero-init per-block modulation refinement so blocks start at shared_mod.
        for block in self.blocks:
            if self.adaln_mode == "adaln_lora":
                nn.init.constant_(block.modulation.lora[-1].weight, 0)
            elif self.adaln_mode == "per_block":
                nn.init.constant_(block.modulation.full[-1].weight, 0)
                nn.init.constant_(block.modulation.full[-1].bias, 0)

        # action stream: zero-init the modulation head so action ramps in.
        if self.action_encoder is not None:
            nn.init.constant_(self.action_encoder.to_mod[-1].weight, 0)
            nn.init.constant_(self.action_encoder.to_mod[-1].bias, 0)

        # pose stream: zero-init so pose modulation ramps in from identity.
        if self.pose_encoder is not None:
            nn.init.constant_(self.pose_encoder.to_mod.weight, 0)
            nn.init.constant_(self.pose_encoder.to_mod.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ------------------------------------------------------------------ #
    def unpatchify(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        c = self.out_channels
        p_t, p_h, p_w = self.x_embedder.patch_size
        t_in, h_in, w_in = self.x_embedder.input_size
        grid_t, grid_h, grid_w = t_in // p_t, h_in // p_h, w_in // p_w
        assert n == grid_t * grid_h * grid_w, f"seq len {n} != grid {grid_t}x{grid_h}x{grid_w}"
        x = x.reshape(b, grid_t, grid_h, grid_w, p_t, p_h, p_w, c)
        x = torch.einsum("nthwpqrc->nctphqwr", x)
        return x.reshape(b, c, grid_t * p_t, grid_h * p_h, grid_w * p_w)

    # ------------------------------------------------------------------ #
    def _resolve_drop_mask(self, b: int, device, cond_drop: Optional[Tensor]) -> Optional[Tensor]:
        """Resolve a per-sample CFG drop mask ``(B,)`` bool, or None.

        Explicit ``cond_drop`` wins; otherwise sample from ``cond_dropout_prob``
        while training.  Shared by the action and pose streams so a dropped
        sample is fully unconditional.
        """
        if cond_drop is not None:
            return cond_drop.to(device=device, dtype=torch.bool).view(b)
        if self.training and self.cond_dropout_prob > 0:
            return torch.rand(b, device=device) < self.cond_dropout_prob
        return None

    def _build_conditioning(
        self,
        t: Tensor,
        y: Optional[Tensor],
        b: int,
        grid_t: int,
        n: int,
        frame_ids: Tensor,
        device,
        dtype,
        cond_drop: Optional[Tensor],
        frame_offset: int = 0,
        framewise: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Build embeddings and modulation, optionally keeping uniform terms compact.

        * ``emb_tok``       : ``(B, N, D)`` timestep(+action) embedding.
        * ``shared_mod_tok``: ``(B, N, 6D)`` shared AdaLN modulation, or None
          (``per_block`` mode does not use it).
        * ``pose_mod_tok``  : ``(B, N, 6D)`` pose modulation, or None.

        With ``framewise=True``, embedding/shared terms keep shape (B, T, D/6D).
        Real pose terms remain (B, N, 6D); per_block action deltas use (B, T, 6D).
        Callers must pass the frame layout to blocks and the final layer too.

        ``frame_offset`` is the absolute temporal index of this window's first
        latent frame (0 for whole-clip / training / the first streaming window;
        >0 for later streaming windows). It gates the ``action_null_first``
        behaviour so only the true global frame 0 is treated as action-free.
        """
        # ---- timestep -> per-token embedding --------------------------- #
        per_token_t = False
        if t.dim() == 1:
            t = t.view(b, 1).expand(b, grid_t)
        elif t.dim() == 2:
            if t.size(1) == n:
                per_token_t = True
            elif t.size(1) != grid_t:
                raise ValueError(f"t shape {t.shape} != frames {grid_t} or tokens {n}")
        else:
            raise ValueError(f"Unsupported timestep shape: {t.shape}")
        if framewise and per_token_t and n != grid_t:
            raise ValueError("Framewise conditioning requires global or per-frame timesteps")
        t_emb = self.t_embedder(t)  # (B, grid_t, D) or (B, N, D)

        drop_mask = self._resolve_drop_mask(b, device, cond_drop)  # (B,) bool or None

        # ---- action -> fold into timestep / modulation stream ---------- #
        if self.action_encoder is not None:
            null = self.null_action.to(device=device, dtype=dtype)
            if y is None:
                y = null.expand(b, grid_t, -1)
            else:
                if y.size(1) == 1:
                    y = y.expand(b, grid_t, -1)
                assert y.size(1) == grid_t, f"action T {y.size(1)} != latent frames {grid_t}"
                if drop_mask is not None:
                    m = drop_mask.view(b, 1, 1).to(y.dtype)
                    y = y * (1 - m) + null * m
            # The true first latent frame is the seed / initial observation and
            # has no preceding action -> use the learned null there. Only when
            # this window actually starts at global frame 0 (frame_offset == 0),
            # so later streaming windows keep their real per-frame actions.
            if self.action_null_first and frame_offset == 0 and grid_t > 0:
                y = y.clone()
                y[:, 0:1, :] = null
            a_emb, a_mod = self.action_encoder(y)  # (B,grid_t,D), (B,grid_t,6D)
            # Token-varying timesteps still need the corresponding frame action.
            # The common per-frame path keeps the original addition unchanged.
            t_emb = t_emb + (a_emb[:, frame_ids, :] if per_token_t else a_emb)
            action_mod = a_mod
        else:
            action_mod = None

        # ---- normalize the combined timestep(+action) embedding -------- #
        # Applied unconditionally (part of the timestep pipeline; also helps the
        # pose / no-action paths). The 6D modulation deltas stay un-normed.
        t_emb = self.emb_norm(t_emb)

        # ---- broadcast per-frame -> per-token -------------------------- #
        emb_tok = t_emb if (per_token_t or framewise) else t_emb[:, frame_ids, :]

        shared_mod_tok: Optional[Tensor] = None
        if self.adaln_mode in ("adaln_lora", "fully_shared"):
            shared = self.shared_mod(emb_tok)  # (B, T, 6D) or (B, N, 6D)
            if action_mod is not None:
                shared = shared + (action_mod if framewise else action_mod[:, frame_ids, :])
            shared_mod_tok = shared
        elif action_mod is not None:
            # per_block mode has no shared term; route action modulation
            # through the pose channel (both are additive per-token deltas).
            if not framewise:
                action_mod = action_mod[:, frame_ids, :]

        # ---- pose -> per-token spatial modulation ---------------------- #
        pose_mod_tok: Optional[Tensor] = None
        if self.pose_encoder is not None and y is not None:
            pose_mod_tok = self.pose_encoder(y, b, grid_t)  # (B, N, 6D)
            if drop_mask is not None:
                # dropped samples become unconditional -> zero pose modulation.
                pose_mod_tok = pose_mod_tok * (~drop_mask).view(b, 1, 1).to(pose_mod_tok.dtype)

        # in per_block mode, fold action delta into the pose channel
        if self.adaln_mode == "per_block" and action_mod is not None:
            pose_mod_tok = action_mod if pose_mod_tok is None else pose_mod_tok + action_mod

        return emb_tok, shared_mod_tok, pose_mod_tok

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        use_fp16: bool = False,
        temporal_causal: bool = False,
        chunk_size: Optional[int] = None,
        cond_drop: Optional[Tensor] = None,
        frame_offset: int = 0,
    ):
        """Forward pass.

        Args:
            x: ``(B, C, T, H, W)`` latent video.
            t: ``(B,)`` / ``(B, T')`` per-frame or ``(B, N)`` per-token timesteps.
            y: condition. If ``cond_per_token`` -> ``(B, T', cond_dim, H, W)``
               ray-encoding; else ``(B, T', cond_dim)`` action.
            cond_drop: optional per-sample bool ``(B,)`` forcing the null / uncond
               condition (for CFG). When None and training, sampled from
               ``cond_dropout_prob``.

        Returns:
            ``v_pred`` of shape ``(B, C, T, H, W)``.
        """
        x = self.x_embedder(x)

        p_t, p_h, p_w = self.x_embedder.patch_size
        t_in, h_in, w_in = self.x_embedder.input_size
        grid_t, grid_h, grid_w = t_in // p_t, h_in // p_h, w_in // p_w
        tokens_per_frame = grid_h * grid_w
        b, n, _ = x.shape
        assert n == grid_t * tokens_per_frame, f"token len {n} != grid {grid_t}x{grid_h}x{grid_w}"
        chunk_size = 1 if chunk_size is None else chunk_size

        attn_mask = None
        if temporal_causal:
            attn_mask = _build_temporal_chunkwise_attn_mask(
                seq_len=n,
                tokens_per_frame=tokens_per_frame,
                device=x.device,
                dtype=x.dtype,
                chunk_size=chunk_size,
            )

        frame_ids = torch.arange(n, device=x.device, dtype=torch.long) // tokens_per_frame
        emb_tok, shared_mod_tok, pose_mod_tok = self._build_conditioning(
            t, y, b, grid_t, n, frame_ids, x.device, x.dtype, cond_drop,
            frame_offset=frame_offset,
        )

        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(
                    block, x, emb_tok, shared_mod_tok, pose_mod_tok, self.feat_rope, attn_mask,
                    use_reentrant=True,
                )
            else:
                x = block(x, emb_tok, shared_mod_tok, pose_mod_tok, self.feat_rope, attn_mask)

        # final layer uses the per-frame(-broadcast) timestep embedding
        x = self.final_layer(x, emb_tok)
        x = self.unpatchify(x)
        return x

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_with_cache(
        self,
        x: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
        past_kv_list: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
        current_position_offset: int = 0,
        return_kv: bool = False,
        chunk_size: int = 1,
        cond_drop: Optional[Tensor] = None,
        use_chunk_flash_attn: Optional[bool] = None,
    ):
        """Streaming forward with optional KV cache injection (RoPE-only).

        Chunk FlashAttention replaces only the cached block-causal attention
        kernel. By default, inherit the model's optimization configuration.
        Explicit ``use_chunk_flash_attn=True/False`` overrides it for this call
        (preserving existing A/B benchmarks); False uses original masked SDPA.
        Cache capacity, sink, chunk boundaries and RoPE are unchanged.
        """
        if use_chunk_flash_attn is None:
            use_chunk_flash_attn = self.optimization_config.chunkwise_flash_attention
        elif type(use_chunk_flash_attn) is not bool:
            raise TypeError("use_chunk_flash_attn must be bool or None")
        b, c_in, t_cur, h_in, w_in = x.shape
        x = self.x_embedder(x)

        p_t, p_h, p_w = self.x_embedder.patch_size
        _, h_total, w_total = self.x_embedder.input_size
        grid_h, grid_w = h_total // p_h, w_total // p_w
        assert h_in == h_total and w_in == w_total, (
            f"forward_with_cache expects {h_total}x{w_total}, got {h_in}x{w_in}"
        )
        grid_t_cur = t_cur // p_t
        tokens_per_frame = grid_h * grid_w
        n_cur = grid_t_cur * tokens_per_frame
        assert x.shape[1] == n_cur, f"patch embed produced {x.shape[1]} tokens, expected {n_cur}"

        if past_kv_list is None:
            past_kv_list = [None] * self.depth
        assert len(past_kv_list) == self.depth

        n_past = 0
        first_past = next((kv for kv in past_kv_list if kv is not None), None)
        if first_past is not None:
            n_past = int(first_past[0].shape[-2])
            assert n_past == current_position_offset * tokens_per_frame, (
                f"past token len {n_past} != offset*tokens_per_frame "
                f"{current_position_offset * tokens_per_frame}"
            )

        # Explicit FlashAttention requires CUDA and FP16/BF16. Keep the original
        # dense mask / SDPA path for FP32, CPU and unsupported head dimensions;
        # never silently lower a caller's FP32 computation to enable the kernel.
        attn_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        use_chunk_flash_attn = (
            use_chunk_flash_attn
            and x.is_cuda
            and attn_dtype in (torch.float16, torch.bfloat16)
            and torch.cuda.get_device_capability(x.device)[0] >= 8
            and all(block.attn.head_dim <= 256 and block.attn.head_dim % 8 == 0 for block in self.blocks)
        )
        cached_chunk_tokens = tokens_per_frame * max(1, int(chunk_size)) if use_chunk_flash_attn else None
        attn_mask = None
        if not use_chunk_flash_attn:
            attn_mask = _build_cached_block_causal_mask(
                n_past=n_past,
                n_cur=n_cur,
                tokens_per_frame=tokens_per_frame,
                chunk_size=max(1, int(chunk_size)),
                device=x.device,
                dtype=x.dtype,
            )

        def feat_rope_current(tt: Tensor) -> Tensor:
            return self.feat_rope(tt, num_frames_override=grid_t_cur, start_frame=int(current_position_offset))

        # Eligibility is structural: never inspect timestep values on the GPU
        # or collapse truly token-varying conditioning. Only this cached eval
        # path opts in; ordinary forward/training retain their original layout.
        # Recompute on each call, including CFG branches and denoising steps.
        framewise = (
            self.optimization_config.framewise_conditioning
            and not self.training
            and (t.dim() == 1 or (t.dim() == 2 and t.shape[1] == grid_t_cur))
        )
        conditioning_tokens_per_frame = tokens_per_frame if framewise else None
        frame_ids = torch.arange(n_cur, device=x.device, dtype=torch.long) // tokens_per_frame
        emb_tok, shared_mod_tok, pose_mod_tok = self._build_conditioning(
            t, y, b, grid_t_cur, n_cur, frame_ids, x.device, x.dtype, cond_drop,
            frame_offset=int(current_position_offset),
            framewise=framewise,
        )

        new_kv_list: List[Optional[Tuple[Tensor, Tensor]]] = [None] * self.depth
        for idx, block in enumerate(self.blocks):
            out = block(
                x, emb_tok, shared_mod_tok, pose_mod_tok, feat_rope_current, attn_mask,
                past_kv_list[idx], return_kv,
                cached_chunk_tokens=cached_chunk_tokens,
                conditioning_tokens_per_frame=conditioning_tokens_per_frame,
            )
            if return_kv:
                x, new_kv_list[idx] = out
            else:
                x = out

        x = self.final_layer(x, emb_tok, conditioning_tokens_per_frame)
        c_out = self.out_channels
        x_vid = x.reshape(b, grid_t_cur, grid_h, grid_w, p_t, p_h, p_w, c_out)
        x_vid = torch.einsum("nthwpqrc->nctphqwr", x_vid)
        v_pred = x_vid.reshape(b, c_out, grid_t_cur * p_t, grid_h * p_h, grid_w * p_w)

        if return_kv:
            return v_pred, new_kv_list
        return v_pred, None


# --------------------------------------------------------------------------- #
#                               Factory configs                               #
# --------------------------------------------------------------------------- #
# Scaling-law ladder. All use patch_size=1; approx param counts are for the
# action mode (cond_dim~128); pose mode differs by only a few tens of M.
#
#   public name  hidden  depth  heads  head_dim   ~params
#   B              768     12     12       64       ~0.12B
#   L             1024     24     16       64       ~0.39B
#   0.5B          1152     28     16       72       ~0.55B
#   1B            1536     28     12      128       ~0.96B
#   3B            2560     32     20      128       ~2.9B
def MiniWorld_B(**kwargs):
    """~0.12B. hidden=768, depth=12, heads=12 (head_dim=64)."""
    return MiniWorldModel(depth=12, hidden_size=768, num_heads=12, patch_size=1, **kwargs)


def MiniWorld_L(**kwargs):
    """~0.39B. hidden=1024, depth=24, heads=16 (head_dim=64)."""
    return MiniWorldModel(depth=24, hidden_size=1024, num_heads=16, patch_size=1, **kwargs)


def MiniWorld_0_5B(**kwargs):
    """~0.55B. hidden=1152, depth=28, heads=16 (head_dim=72)."""
    return MiniWorldModel(depth=28, hidden_size=1152, num_heads=16, patch_size=1, **kwargs)


def MiniWorld_1B(**kwargs):
    """~0.96B. hidden=1536, depth=28, heads=12 (head_dim=128)."""
    return MiniWorldModel(depth=28, hidden_size=1536, num_heads=12, patch_size=1, **kwargs)


def MiniWorld_3B(**kwargs):
    """~2.9B. hidden=2560, depth=32, heads=20 (head_dim=128)."""
    return MiniWorldModel(depth=32, hidden_size=2560, num_heads=20, patch_size=1, **kwargs)


MiniWorldModels = {
    "B": MiniWorld_B,
    "L": MiniWorld_L,
    "0.5B": MiniWorld_0_5B,
    "1B": MiniWorld_1B,
    "3B": MiniWorld_3B,
}
