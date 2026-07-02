"""Python smoke test for the `turboquant` extension module.

Not run in CI (cargo ignores .py files under tests/). To run it locally:

    pip install maturin numpy
    maturin develop -m rust/crates/turboquant-py/Cargo.toml
    python rust/crates/turboquant-py/tests/smoke.py
"""

import numpy as np

import turboquant


def main() -> None:
    assert isinstance(turboquant.__version__, str) and turboquant.__version__

    # Quantize / dequantize round-trip with correction.
    q = turboquant.Quantizer(bits=3, block_size=64, scale_mode="absmax", correction=True)
    data = ((np.arange(64, dtype=np.float32) - 32.0) / 10.0).astype(np.float32)
    packed, scale, correction = q.quantize(data)
    assert packed.dtype == np.uint8 and packed.shape == (24,)
    assert isinstance(scale, float) and scale > 0.0
    assert correction is not None and correction.dtype == np.uint8 and correction.shape == (8,)

    restored = q.dequantize(packed, 64, scale, correction)
    assert restored.dtype == np.float32 and restored.shape == (64,)
    mse = float(np.mean((data - restored) ** 2))
    assert mse < 0.25, f"mse too high: {mse}"

    # Correction disabled -> correction is None.
    q2 = turboquant.Quantizer(correction=False)
    _, _, corr2 = q2.quantize(data)
    assert corr2 is None

    # pack_3bit / unpack_3bit round-trip.
    values = (np.arange(64, dtype=np.uint8) % 8).astype(np.uint8)
    packed_bits = turboquant.pack_3bit(values)
    assert packed_bits.shape == (24,)
    assert np.array_equal(turboquant.unpack_3bit(packed_bits, 64), values)

    # hadamard_rotate forward + inverse round-trip.
    x = np.sin(np.arange(128, dtype=np.float32))
    y = turboquant.hadamard_rotate(x, seed=42)
    assert not np.allclose(x, y)
    x2 = turboquant.hadamard_rotate(y, seed=42, inverse=True)
    assert np.allclose(x, x2, atol=1e-4)

    # Errors surface as ValueError.
    for bad_call in (
        lambda: turboquant.Quantizer(bits=4),
        lambda: turboquant.Quantizer(scale_mode="bogus"),
        lambda: turboquant.Quantizer(scale_mode="percentile"),  # missing percentile
        lambda: q.quantize(np.empty(0, dtype=np.float32)),
        lambda: q.dequantize(packed, 64, -1.0, correction),
        lambda: turboquant.pack_3bit(np.full(8, 9, dtype=np.uint8)),
        lambda: turboquant.hadamard_rotate(np.zeros(3, dtype=np.float32), seed=1),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError from {bad_call}")

    print("python smoke ok:", turboquant.__version__, "mse =", mse)


if __name__ == "__main__":
    main()
