"""Override: requantize the MTP layer's GGUF projections (Q6_K in the GSQ-RCO release) to IQ4_XS.

The MTP layer is read once per draft step (4x per round at --draft-tokens 4); Q6_K costs 6.56
bits/weight, IQ4_XS 4.25 and runs through the same ggml vector kernels as the text layers. The
source values are the GGUF's dequantized rows, so this is a requantization. The quantizer follows
ggml's quantize_row_iq4_nl_impl for IQ4_XS (256-value superblocks, 32-value blocks, ntry 7,
weights x^2 without an importance matrix); the superblock scale is rounded to FP16 before the block
scales are chosen.

Use it after the qwen3_8_27b_gguf recipe:

    python -m tools.convert --recipe qwen3_8_27b_gguf         --override tools/convert/overrides/qwen3_8_mtp_iq4xs.py ...

scripts/qwen38/prepare-model.sh runs the full conversion.
"""

import numpy as np
import torch

from tools.convert.gguf_blocks import text_sources
from tools.convert.methods import import_encoded
from tools.convert.sources.logical import EncodedRows, LogicalSource

FORMAT = "gguf_iq4_xs"
VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32
)
MIDPOINTS = (VALUES[1:] + VALUES[:-1]) / 2
NTRY = 7
BLOCK_BYTES = 136


def _nearest(v: np.ndarray) -> np.ndarray:
    # ggml best_index_int8: the nearest grid value, ties to the upper one.
    return np.searchsorted(MIDPOINTS, v, side="right").astype(np.uint8)


def _fit(x: np.ndarray, w: np.ndarray, inverse: np.ndarray):
    """Codes for x * inverse (per block) and their weighted least-squares numerator/denominator."""
    q = VALUES[_nearest(x * inverse[..., None])]
    return (w * q * x).sum(-1), (w * q * q).sum(-1)


def quantize_iq4_xs(rows: np.ndarray) -> np.ndarray:
    """uint8 [n, k/256*136] IQ4_XS blocks of float32 rows [n, k], weighting the error by x^2."""
    n, k = rows.shape
    x = rows.astype(np.float32).reshape(-1, 8, 32)
    w = x * x
    a = np.abs(x)
    arg = a.argmax(-1)
    amax = np.take_along_axis(a, arg[..., None], -1)[..., 0]
    signed = np.take_along_axis(x, arg[..., None], -1)[..., 0]
    live = amax >= 1e-15
    safe = np.where(live, signed, 1.0)
    d = -safe / VALUES[0]
    sumqx, sumq2 = _fit(x, w, 1.0 / d)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sumq2 > 0, sumqx / sumq2, 0.0)
    best = d * sumqx
    for itry in range(-NTRY, NTRY + 1):
        sumqx, sumq2 = _fit(x, w, (itry + VALUES[0]) / safe)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(better, sumqx / sumq2, d)
        best = np.where(better, d * sumqx, best)
    scales = np.where(live, d, 0.0).astype(np.float32)
    pick = np.abs(scales).argmax(-1)
    max_scale = np.take_along_axis(scales, pick[..., None], -1)[..., 0]
    dh = (-max_scale / 32).astype(np.float16)
    dsb = dh.astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(dsb != 0, 1.0 / dsb, 0.0)
    level = np.clip(np.rint(inv[:, None] * scales), -32, 31).astype(np.int32)
    dl = dsb[:, None] * level
    with np.errstate(divide="ignore"):
        idl = np.where(dl != 0, 1.0 / dl, 0.0)
    codes = _nearest(x * idl[..., None])
    stored = (level + 32).astype(np.uint32)
    out = np.zeros((x.shape[0], BLOCK_BYTES), np.uint8)
    out[:, 0:2] = dh.view(np.uint8).reshape(-1, 2)
    high = np.zeros(x.shape[0], np.uint32)
    for ib in range(8):
        high |= (stored[:, ib] >> 4) << (2 * ib)
    out[:, 2:4] = high.astype("<u2").view(np.uint8).reshape(-1, 2)
    low = stored & 0xF
    out[:, 4:8] = (low[:, 0::2] | (low[:, 1::2] << 4)).astype(np.uint8)
    out[:, 8:] = (codes[:, :, :16] | (codes[:, :, 16:] << 4)).reshape(-1, 128)
    return out.reshape(n, k // 256 * BLOCK_BYTES)


def dequantize_iq4_xs(blocks: np.ndarray, k: int) -> np.ndarray:
    b = blocks.reshape(-1, BLOCK_BYTES)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    high = b[:, 2:4].copy().view("<u2").astype(np.uint32)[:, 0]
    low = np.stack([b[:, 4:8] & 0xF, b[:, 4:8] >> 4], -1).reshape(-1, 8).astype(np.int32)
    ls = low | (((high[:, None] >> (2 * np.arange(8))) & 3) << 4).astype(np.int32)
    q = b[:, 8:].reshape(-1, 8, 16)
    codes = np.concatenate([q & 0xF, q >> 4], -1)
    values = (d[:, None, None] * (ls - 32)[..., None]) * VALUES[codes]
    return values.reshape(-1, k)


def iq4_source(source: LogicalSource) -> LogicalSource:
    n, k = source.shape
    if k % 256:
        raise ValueError(f"{source.label}: K={k} is not a multiple of 256")

    def encode(begin: int, end: int) -> np.ndarray:
        values = source.values(begin * k, end * k).reshape(end - begin, k)
        return quantize_iq4_xs(values.to(torch.float32).numpy())

    def encoded(begin: int, end: int) -> EncodedRows:
        return EncodedRows(
            FORMAT, torch.from_numpy(encode(begin, end)), torch.empty(0, dtype=torch.float16)
        )

    def read_values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        flat = dequantize_iq4_xs(encode(first, last), k).reshape(-1)
        return torch.from_numpy(flat[begin - first * k : end - first * k].copy())

    return LogicalSource(source.shape, f"{source.label}->{FORMAT}", read_values, encoded)


def configure(model, recipe, sources):
    encoded, _ = text_sources(sources["gguf"], mtp=True)
    for name, source in encoded.items():
        if name.startswith("mtp/"):
            recipe.assign(name, format=FORMAT, method=import_encoded, source=iq4_source(source))
