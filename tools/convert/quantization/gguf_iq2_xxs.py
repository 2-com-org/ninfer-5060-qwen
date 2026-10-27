"""GGUF IQ2_XXS blocks (2.0625 bits per weight) of float rows.

A 256-value superblock stores an FP16 scale d and eight 32-value sub-blocks. A sub-block holds four
8-value groups, each one byte indexing ggml's iq2xxs grid of 256 magnitude patterns (values 8, 25
and 43) and seven sign bits, the eighth making the group's count of negatives even; its 4-bit
scale ls gives every value of the sub-block d * (0.5 + ls) / 4 * grid * sign. The layout is
ggml's block_iq2_xxs: d, then per sub-block the four grid bytes and a word of the four 7-bit sign
indices (bits 0, 7, 14, 21) with ls in bits 28..31.

The encoder is not ggml's quantize_row_iq2_xxs: per sub-block it tries a few scales (the largest
magnitude at grid level 43, 36, 30 or 25), takes the nearest grid pattern of each group's
magnitudes, rounds the sub-block scales to the superblock's 4-bit steps and assigns the groups
again at the stored scale; an odd sign count flips the element whose sign costs least. It serves
coarse proposal heads, whose quality is their top-K recall of the exact head's argmax.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.convert.sources.logical import EncodedRows, LogicalSource

FORMAT = "gguf_iq2_xxs"
BLOCK_VALUES = 256
BLOCK_BYTES = 66
_LEVELS = (43.0, 36.0, 30.0, 25.0)


def _grid() -> torch.Tensor:
    # gguf-py carries ggml's tables; the GGUF sources already require the package.
    from gguf import quants

    quants.IQ2_XXS.init_grid()
    return torch.from_numpy(quants.IQ2_XXS.grid.reshape(256, 8).astype(np.float32))


def quantize_iq2_xxs(rows: torch.Tensor, chunk: int = 256) -> np.ndarray:
    """uint8 [n, k / 256 * 66] IQ2_XXS blocks of rows [n, k] (k a multiple of 256), `chunk` rows
    at a time (the grid search holds 256 distances per 8 values)."""
    n, k = rows.shape
    if k % BLOCK_VALUES:
        raise ValueError(f"IQ2_XXS rows need a multiple of {BLOCK_VALUES} values, got {k}")
    grid = _grid()
    out = np.empty((n, k // 256 * BLOCK_BYTES), dtype=np.uint8)
    for first in range(0, n, chunk):
        out[first : first + chunk] = _quantize(rows[first : first + chunk], grid)
    return out


def _quantize(rows: torch.Tensor, grid: torch.Tensor) -> np.ndarray:
    x = rows.to(torch.float32)
    n, k = x.shape
    grid_sq = (grid * grid).sum(1)

    def assign(u: torch.Tensor) -> torch.Tensor:
        # Nearest grid pattern (L2) of each group of eight magnitudes in grid units.
        return (grid_sq.unsqueeze(0) - 2.0 * (u @ grid.T)).argmin(1)

    sub = x.reshape(n, k // 32, 32)
    mag = sub.abs()
    amax = mag.amax(2)
    best_err = best_sigma = None
    for level in _LEVELS:
        sigma = (amax / level).clamp_min(torch.finfo(torch.float32).tiny)
        g = grid[assign((mag / sigma.unsqueeze(2)).reshape(-1, 8))]
        err = ((mag - g.reshape(n, k // 32, 32) * sigma.unsqueeze(2)) ** 2).sum(2)
        if best_err is None:
            best_err, best_sigma = err, sigma
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_sigma = torch.where(better, sigma, best_sigma)
    sig = best_sigma.reshape(n, k // 256, 8)
    dh = (sig.amax(2) / (15.5 * 0.25)).clamp(max=65504.0).half()
    d = dh.float()
    live = d > 0
    safe = torch.where(live, d, torch.ones_like(d))
    ls = torch.clamp(torch.round(sig / (safe.unsqueeze(2) * 0.25) - 0.5), 0, 15)
    ls = torch.where(live.unsqueeze(2), ls, torch.zeros_like(ls))
    sigma = (safe.unsqueeze(2) * (0.5 + ls) * 0.25).reshape(n, k // 32)
    idx = assign((mag / sigma.unsqueeze(2)).reshape(-1, 8)).reshape(n, k // 32, 4)
    g = grid[idx]
    xs = sub.reshape(n, k // 32, 4, 8)
    neg = xs < 0
    odd = (neg.sum(3) % 2) == 1
    flip = torch.zeros_like(neg)
    flip.scatter_(3, (xs.abs() * g).argmin(3, keepdim=True), True)
    neg = neg ^ (flip & odd.unsqueeze(3))
    bits = (neg[..., :7].to(torch.int64) << torch.arange(7)).sum(3)
    aux = (bits[..., 0] | (bits[..., 1] << 7) | (bits[..., 2] << 14) | (bits[..., 3] << 21)
           | (ls.reshape(n, k // 32).to(torch.int64) << 28))
    q0 = idx.to(torch.int64)
    q0 = q0[..., 0] | (q0[..., 1] << 8) | (q0[..., 2] << 16) | (q0[..., 3] << 24)
    words = torch.stack((q0, aux), 2).reshape(n, k // 256, 16).numpy().astype("<u4")
    scale = dh.numpy().astype("<f2").view(np.uint8).reshape(n, k // 256, 2)
    blocks = np.concatenate((scale, words.view(np.uint8).reshape(n, k // 256, 64)), 2)
    return np.ascontiguousarray(blocks.reshape(n, k // 256 * BLOCK_BYTES))


def dequantize_iq2_xxs(blocks: np.ndarray, k: int) -> np.ndarray:
    """float32 [n, k] values of IQ2_XXS blocks [n, k / 256 * 66]."""
    from gguf import GGMLQuantizationType, quants

    flat = quants.dequantize(np.ascontiguousarray(blocks), GGMLQuantizationType.IQ2_XXS)
    return flat.reshape(-1, k).astype(np.float32)


def iq2_xxs_source(source: LogicalSource) -> LogicalSource:
    """IQ2_XXS rows of a matrix source, encoded from its values."""
    n, k = source.shape
    if k % BLOCK_VALUES:
        raise ValueError(f"{source.label}: K={k} is not a multiple of {BLOCK_VALUES}")

    def encode(begin: int, end: int) -> np.ndarray:
        return quantize_iq2_xxs(source.rows(begin, end))

    def encoded(begin: int, end: int) -> EncodedRows:
        return EncodedRows(
            FORMAT, torch.from_numpy(encode(begin, end)), torch.empty(0, dtype=torch.float16)
        )

    def read_values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        flat = dequantize_iq2_xxs(encode(first, last), k).reshape(-1)
        return torch.from_numpy(flat[begin - first * k : end - first * k].copy())

    return LogicalSource(source.shape, f"{source.label}->{FORMAT}", read_values, encoded)


__all__ = ["FORMAT", "dequantize_iq2_xxs", "iq2_xxs_source", "quantize_iq2_xxs"]
