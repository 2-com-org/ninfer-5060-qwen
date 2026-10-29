from __future__ import annotations

import numpy as np
import pytest

from tools.convert.overrides import qwen3_8_mtp_iq4xs as iq4

gguf = pytest.importorskip("gguf")


def _rows(seed: int, rows: int, columns: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((rows, columns)).astype(np.float32) * 0.02
    # Outliers and a silent block exercise the scale search and the zero-block path.
    values[0, 5] = 0.4
    values[1, 32:64] = 0.0
    return values


def test_blocks_decode_exactly_as_gguf_py():
    values = _rows(1, 8, 1024)
    blocks = iq4.quantize_iq4_xs(values)
    assert blocks.shape == (8, 1024 // 256 * iq4.BLOCK_BYTES)
    reference = gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.IQ4_XS).reshape(8, 1024)
    np.testing.assert_array_equal(iq4.dequantize_iq4_xs(blocks, 1024), reference)


def test_requantization_error_is_bounded():
    values = _rows(2, 16, 5120)
    decoded = gguf.quants.dequantize(iq4.quantize_iq4_xs(values),
                                     gguf.GGMLQuantizationType.IQ4_XS).reshape(values.shape)
    relative = np.sqrt(((decoded - values) ** 2).mean() / (values ** 2).mean())
    # 4.25 bits per weight on Gaussian rows; the real MTP matrices measured 0.077-0.086.
    assert relative < 0.1
    np.testing.assert_array_equal(decoded[1, 32:64], 0.0)


def test_all_zero_rows_encode_to_zero():
    blocks = iq4.quantize_iq4_xs(np.zeros((2, 256), np.float32))
    np.testing.assert_array_equal(iq4.dequantize_iq4_xs(blocks, 256), 0.0)
