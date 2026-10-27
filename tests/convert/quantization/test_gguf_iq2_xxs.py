import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
from gguf import quants  # noqa: E402

from tools.convert.quantization.gguf_iq2_xxs import (  # noqa: E402
    BLOCK_BYTES,
    dequantize_iq2_xxs,
    iq2_xxs_source,
    quantize_iq2_xxs,
)
from tools.convert.sources.logical import array_source  # noqa: E402


def _representable(rng, rows, k):
    """Rows the format holds exactly: every sub-block at ls 15 or below with one value at grid
    level 43 (so the encoder's scale search meets its true scale), signs of even parity."""
    quants.IQ2_XXS.init_grid()
    grid = quants.IQ2_XXS.grid.reshape(256, 8).astype(np.float32)
    peak = np.flatnonzero((grid == 43).any(1))
    out = np.empty((rows, k), np.float32)
    for r in range(rows):
        for b in range(k // 256):
            d = np.float32(np.float16(rng.uniform(1e-4, 1e-2)))
            ls = rng.integers(0, 16, 8)
            ls[rng.integers(0, 8)] = 15
            for s in range(8):
                sigma = d * (0.5 + ls[s]) * 0.25
                idx = rng.integers(0, 256, 4)
                idx[rng.integers(0, 4)] = rng.choice(peak)
                for g in range(4):
                    signs = rng.integers(0, 2, 8)
                    signs[7] = signs[:7].sum() % 2
                    values = sigma * grid[idx[g]] * np.where(signs == 1, -1.0, 1.0)
                    at = b * 256 + s * 32 + g * 8
                    out[r, at : at + 8] = values
    return out


def test_representable_rows_round_trip_exactly_through_ggml_decode():
    rng = np.random.default_rng(3)
    rows = _representable(rng, 4, 512)
    blocks = quantize_iq2_xxs(torch.from_numpy(rows))
    assert blocks.shape == (4, 2 * BLOCK_BYTES)
    np.testing.assert_array_equal(dequantize_iq2_xxs(blocks, 512), rows)


def test_rows_decode_close_and_zero_rows_stay_zero():
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((6, 1024)).astype(np.float32) * 0.02
    rows[2] = 0.0
    blocks = quantize_iq2_xxs(torch.from_numpy(rows), chunk=4)
    decoded = dequantize_iq2_xxs(blocks, 1024)
    assert not decoded[2].any()
    live = np.delete(np.arange(6), 2)
    err = np.linalg.norm(decoded[live] - rows[live]) / np.linalg.norm(rows[live])
    assert err < 0.45
    # Rows are encoded independently: chunking changes no byte.
    np.testing.assert_array_equal(blocks, quantize_iq2_xxs(torch.from_numpy(rows), chunk=256))


def test_source_encodes_rows_and_reads_their_decoded_values():
    rng = np.random.default_rng(7)
    values = torch.from_numpy(rng.standard_normal((5, 256)).astype(np.float32))
    source = iq2_xxs_source(array_source(values, "rows"))
    encoded = source.read_encoded(1, 4)
    assert encoded.format == "gguf_iq2_xxs" and tuple(encoded.codes.shape) == (3, BLOCK_BYTES)
    np.testing.assert_array_equal(encoded.codes.numpy(), quantize_iq2_xxs(values[1:4]))
    flat = dequantize_iq2_xxs(quantize_iq2_xxs(values), 256).reshape(-1)
    np.testing.assert_array_equal(source.values(300, 900).numpy(), flat[300:900])


def test_rows_need_whole_superblocks():
    with pytest.raises(ValueError):
        quantize_iq2_xxs(torch.zeros(2, 128))
