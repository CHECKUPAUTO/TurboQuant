# Bugs Fixed — TurboQuant Migration to Rust

This document catalogues every defect in the original Python implementation
(`legacy/python/turboquant.py`, 490 lines) and how each is resolved in the
Rust workspace under `rust/`. The Python version is kept intact under
`legacy/python/` for reference only.

---

## Bug 1: Bit-packing is stubbed — no actual 3-bit compression

**Severity**: CRITICAL — makes compression ratio claims completely bogus.

**Python code** (`legacy/python/turboquant.py`, `_pack_bits` / `_unpack_bits`):
```python
# _pack_bits does: (x * 2^(bits-1)).to(int8).view(uint8)
# This is 8-bit storage, not 3-bit. Every value consumes 8 bits.
# No actual packing happens.
```

**Root cause**: The `_pack_bits` method truncates to int8 and reinterprets as
uint8 without any bit-level compaction. For `bits=3`, each value still
occupies a full byte. The claimed ~6× compression ratio is entirely
theoretical.

**Rust fix**: `turboquant_core::bitpack::pack_3bit` / `unpack_3bit`
- Pack exactly 8 × 3-bit values into 24 bits (3 bytes).
- API: `pub fn pack_3bit(values: &[u8], out: &mut [u8])`
- API: `pub fn unpack_3bit(packed: &[u8], out: &mut [u8])`
- Roundtrip property-tested via `proptest` on `0..=4096` element arrays.
- Optional SIMD path behind `simd` feature flag.

**Test**: `tests/bitpack_roundtrip.rs` — proptest roundtrip, edge cases.

---

## Bug 2: `_unpack_bits` symmetric to Bug 1

**Severity**: CRITICAL — companion to Bug 1.

**Python code**: Same root cause — `_unpack_bits` operates on 8-bit values,
not 3-bit packed data.

**Rust fix**: Same as Bug 1 — `unpack_3bit` is the symmetric unpack.

---

## Bug 3: `store(layer_idx, positions, k, v)` ignores `positions`

**Severity**: HIGH — incremental KV cache writes are impossible.

**Python code**:
```python
def store(self, layer_idx, positions, k, v):
    # positions argument is completely ignored
    self.cache_k[str(layer_idx)].data = k_packed  # overwrites entire cache
```

**Root cause**: `store` overwrites the entire cache for a given layer
regardless of which positions are being written. When a new token is
generated, the entire accumulated cache is replaced by the single new token's
K/V values.

**Rust fix**: `KvBlock::store(layer_idx, range: Range<usize>, k, v)`
- Writes at the exact byte offset in packed memory.
- Validates alignment to 8-value boundaries for 3-bit packing.
- Returns `Result` with error on misaligned ranges.

**Test**: `tests/kv_block_store_partial.rs` — write 3 consecutive ranges,
verify all ranges readable.

---

## Bug 4: `retrieve()` ignores `positions` and averages scales

**Severity**: HIGH — retrieved values are garbage.

**Python code**:
```python
def retrieve(self, layer_idx, positions):
    # positions argument is ignored
    return scales_k[positions].mean()  # averaging destroys per-token info
```

**Root cause**: Instead of indexing the exact compressed tokens, the method
averages scales across all positions. The per-token information structure
is destroyed.

**Rust fix**: `KvBlock::retrieve(layer_idx, positions: &[usize])`
- Exact per-block scale lookup.
- Returns decompressed values for the requested positions only.

**Test**: `tests/kv_block_retrieve_exact.rs` — compress 64 tokens, retrieve
positions [0, 1, 10, 63], verify numeric agreement.

---

## Bug 5: `nn.Parameter(torch.zeros(..., dtype=torch.uint8))` is invalid

**Severity**: MEDIUM — crashes at `nn.Parameter` creation.

**Python code**:
```python
self.data = nn.Parameter(torch.zeros(size, dtype=torch.uint8))
```

**Root cause**: `nn.Parameter` requires a float or complex tensor. `uint8`
is not a valid dtype for parameter creation.

**Rust fix**: Plain `Vec<u8>` for packed data, `Vec<f16>` for per-block
scales. No ML framework dependency imposed. The storage is a simple struct:
```rust
pub struct KvBlock {
    pub data: Vec<u8>,        // packed 3-bit values
    pub scales: Vec<f16>,     // one f16 scale per block
    pub block_size: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}
```

**Test**: `tests/kv_block_create.rs` — construct and read back.

---

## Bug 6: Scale QJL hardcoded to 0.01 with no derivation

**Severity**: MEDIUM — suboptimal quantization quality.

**Python code**:
```python
self.scale = nn.Parameter(torch.tensor(0.01))
```

**Root cause**: The magic number 0.01 is never justified mathematically.
For many input distributions, this yields excessive quantization error.

**Rust fix**: `QjlConfig` with configurable scale mode:
```rust
pub enum ScaleMode {
    /// Adaptive: derive optimal scale from residual variance per block.
    Adaptive,
    /// Fixed: use a predetermined scale value.
    Fixed(f32),
}
```
Adaptive mode computes `scale = sqrt(variance(residual)) * correction_factor`
where `correction_factor` is derived from the QJL lemma bound.

**Test**: `tests/qjl_scale_optimality.rs` — verify adaptive scale yields
lower reconstruction error than arbitrary fixed scales on random tensors.

---

## Bug 7: `TurboQuantAttention.forward` never uses decompressed K/V

**Severity**: CRITICAL — compression is purely cosmetic.

**Python code**:
```python
def forward(self, hidden_states, attention_mask=None, use_cache=True,
            past_key_value=None):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    # ... cache compresses k/v for storage ...
    # BUT attention is computed on original k, v, not decompressed!
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
```

**Root cause**: The forward pass compresses K and V into the cache (side
effect), but the attention computation uses the uncompressed tensors. The
compression path is never exercised in inference quality.

**Rust fix**: Reference forward that actually decompresses before `Q·Kᵀ`:
```rust
pub fn turbo_attention_forward(q, k_cache, v_cache, mask, out) -> Result<AttentionStats>
```
This decompresses K and V from the cache, then computes attention. The
test verifies that the error on attention output is bounded (SNR > 12 dB).

**Test**: `tests/attention_parity.rs` — compare turbo vs FP16 attention
output, assert SNR > 12 dB and cosine > 0.96.

---

## Bug 8: Scale is global `x.abs().max()` instead of per-block

**Severity**: MEDIUM — outliers dominate the scale for the entire tensor.

**Python code**:
```python
x_max = x.abs().max() + 1e-8
x_scaled = (x / x_max) * half_range
```

**Root cause**: A single outlier (e.g., activation spike in one dimension
at one position) sets the scale for the entire tensor. Most values get
squeezed into near-zero, losing precision.

**Rust fix**: Per-block scaling with configurable block size (64 or 128):
- Each block gets its own `f16` scale.
- Outliers are isolated to their own blocks.
- `ScaleMode::PerBlockPercentile(p99.5)` further clips extreme outliers.

**Test**: `tests/quantize_per_block.rs` — compare per-block vs global scale
error on tensors with injected outliers.

---

## Bug 9: `R @ x` vs `x @ R` confusion between doc and code

**Severity**: LOW — but breaks mathematical understanding.

**Doc says**: `y = R · x` (rotation applied on the left).
**Code does**: `torch.matmul(x, R)` (rotation applied on the right).

**Root cause**: Inconsistent convention between documentation and
implementation. For row-vector convention `(batch, dim)`, `x @ R` is
correct (right-multiply). The doc implies left-multiply.

**Rust fix**: Single documented convention: `y = x · R` (right-multiply,
row vectors). Documented at the trait level. Tests verify orthogonality:
`R · Rᵀ = I` to 1e-5 tolerance.

**Test**: `tests/rotation_convention.rs` — verify `R * Rᵀ ≈ I` and
`‖x·R‖₂ = ‖x‖₂`.

---

## Bug 10: `compression_ratio_vs_fp16` ignores scale overhead

**Severity**: MEDIUM — claims overstate actual compression.

**Python code**:
```python
def compression_ratio_vs_fp16(self):
    return 16 / self.bits  # naive: just bits-per-value ratio
```

**Root cause**: Per-block scales (f16 per 64 or 128 values) add overhead
that reduces the effective compression ratio. The real ratio is lower.

**Rust fix**:
```rust
pub fn compression_ratio_vs_fp16(&self) -> f64 {
    let bits_payload = self.head_dim * self.seq_len * 3;
    let bits_scales = (self.seq_len / self.block_size) * 16;
    16.0 * (self.head_dim * self.seq_len) as f64
        / (bits_payload + bits_scales) as f64
}
```

**Test**: `tests/compression_ratio_includes_overhead.rs` — verify ratio <
naive 16/3 for all realistic parameters.

---

## Bug 11: QR-based orthogonal matrix generation is O(d³)

**Severity**: MEDIUM — prohibitive for large head dimensions.

**Python code**:
```python
H = torch.randn(dim, dim, device=self.device)
Q, _ = torch.linalg.qr(H)
self.R = Q  # O(d³), memory O(d²)
```

**Root cause**: QR decomposition is cubic in `d`. For `head_dim=128`, this
is ~2M operations. For `head_dim=256`, ~16M. The matrix storage is also
O(d²).

**Rust fix**: Three rotation strategies, benched against each other:
- `QrRotation` (baseline, O(d²) apply, O(d³) init, O(d²) storage)
- `HouseholderRotation` (k reflectors, O(k·d) apply, O(k·d) storage)
- `FastHadamardRotation` (O(d log d) apply via FHT, O(d) storage)

**Bench**: `benches/rotation_bench.rs` — criterion comparing init time,
apply time, memory for head_dim ∈ {64, 128, 256}.

---

## Bug 12: `learn_scale=True` exposed but quantization is non-differentiable

**Severity**: LOW — misleading API.

**Python code**:
```python
def __init__(self, bits: int = 3, learn_scale: bool = True):
    if learn_scale:
        self.scale = nn.Parameter(torch.tensor(0.01))
```

**Root cause**: The `round()` and `sign()` operations in quantize break
gradient flow. Setting `learn_scale=True` is meaningless — the scale
cannot be learned through backprop.

**Rust fix**: No "learnable" pretense. Calibration is offline via
`turboquant calibrate` command that processes representative data
passes to determine optimal scale parameters.

**Test**: `tests/calibrate_yaml_output.rs` — run calibrate, verify output
YAML has valid scale values.

---

## Bug 13: No runnable tests or benchmarks

**Severity**: HIGH — no way to verify correctness or measure performance.

**Python code**:
```python
def benchmark_turboquant(...):
    # Only computes theoretical memory, no actual compression/decompression timing
    fp16_bytes = seq_len * head_dim * num_heads * num_layers * 2
    return { 'fp16_memory_mb': fp16_bytes / (1024 * 1024), ... }
```

**Root cause**: The benchmark function is purely arithmetic — no data is
ever compressed or decompressed. No timing measurements.

**Rust fix**: Full criterion benchmark suite:
- Compression throughput (GB/s)
- Decompression throughput (GB/s)
- Dot-product error vs FP16
- Memory usage (actual allocation measurement)
- Realistic sizes: head_dim ∈ {64, 128}, seq_len ∈ {2048, 8192, 16384, 32768}

**Bench**: `benches/compression.rs` — criterion benchmarks.

---

## Additional Issues Discovered During Audit

### Bug 14: `compression_ratio` field is incorrectly named `compression_ration`

**Python code** (typo in `TurboQuantKVCache`):
```python
self.compression_ration = ...  # "ration" instead of "ratio"
```

**Rust fix**: Correctly spelled `compression_ratio` field.

---

### Bug 15: No `#[cfg(test)]` isolation — benchmark code pollutes production paths

The Python `if __name__ == '__main__'` block is fine, but the benchmark
function is always available and imports `torch` unconditionally even when
not benchmarking.

**Rust fix**: Benchmarks in separate `turboquant-bench` crate.

---

### Bug 16: Memory estimates use `sys.getsizeof` which doesn't measure GPU memory

**Python code**: Uses Python-level `sys.getsizeof` or arithmetic on
theoretical sizes. No CUDA memory query (e.g., `torch.cuda.memory_allocated()`).

**Rust fix**: Optional CUDA backend queries actual device memory via
`cudarc` API.

---

## Bugs Introduced (and Fixed) in the Rust Port — 2026-07 audit

The Rust port itself shipped with new defects. These were found and fixed
during the 2026-07 audit; each entry lists the regression test that now
guards it.

### Bug R1: QJL 3-bit grid collapse — every positive value decoded to 0.0

**Severity**: CRITICAL — the quantizer destroyed half of its input.

**Rust code** (`turboquant-core/src/qjl.rs`, `quantize_block`): the code
mapped `x_norm ∈ [-1, 1]` onto a 15-level half-step grid
(`increment = 0.5`, `half_range = 3.5`) and then clamped the resulting
index to `[0, 7]`. Every positive input saturated at index 7, which the
dequantizer decoded as `0.0`. Measured round-trip SNR on Gaussian data:
**2.9 dB** (i.e., the "compressed" signal was mostly noise).

**Fix**: proper symmetric 8-level 3-bit grid
`{-3.5, -2.5, -1.5, -0.5, +0.5, +1.5, +2.5, +3.5}` (spacing 1.0):
`idx = clamp(round(x_norm·3.5 + 3.5), 0, 7)`, dequant
`(idx − 3.5)/3.5 · scale`. Measured SNR after fix: **~13 dB** without
correction, **~19 dB** with the default 1-bit correction.

**Tests**: `src/qjl.rs` — `test_positive_values_roundtrip_positive`,
`test_quantize_dequantize_with_correction` (threshold raised from the
bug-masking `> 2 dB` to `> 12 dB`), `test_dot_product_preserved`.

### Bug R2: Residual sign computed in mixed units

**Severity**: HIGH — the 1-bit correction pushed values the wrong way.

`residual = x_norm - q_clamped` compared a normalized value (`[-1, 1]`)
against a level-unit value (`[-3.5, 3.5]`), so for positive inputs the
correction sign was always negative. Fixed by computing the residual in
level units: `residual = x_norm·3.5 − (idx − 3.5)`.

**Test**: `src/qjl.rs` — `test_correction_improves_snr`.

### Bug R3: 1-bit correction default too small (0.01)

**Severity**: MEDIUM — with the old code, enabling the correction made
reconstruction *worse* than no correction (measured 2.71 dB vs 2.88 dB).

With the fixed grid, the residual is ~uniform in ±0.5 level units, so the
MSE-optimal fixed 1-bit correction is a quarter step: 0.25 level units =
`0.25 / 3.5 ≈ 0.0714` in normalized units. That is now the documented
default (`QjlConfig::default()`); the field remains configurable.

**Test**: `src/qjl.rs` — `test_correction_improves_snr`.

### Bug R4: Multi-block position collision in the tensor pipeline

**Severity**: CRITICAL — silent data corruption for `head_dim > block_size`.

`compress_tensor` stored block `b` of head `h` at position `h + b`
(colliding across heads: head 1's first block landed on head 0's second),
with per-position byte/scale counts that did not match what
`KvBlock::store` expected, and `decompress_tensor` used yet another
indexing. `KvBlock`'s scale indexing (`pos · head_dim / block_size`) also
collided for positions with multiple blocks.

**Fix**: one `KvBlock` position per input row (contiguous packed bytes +
`head_dim.div_ceil(block_size)` scales per position); `KvBlock::store` /
`retrieve_scales` now index scales by `scales_per_position()` and validate
input lengths instead of panicking.

**Tests**: `src/quantize.rs` —
`test_multi_block_positions_do_not_collide` (head_dim = 2×block_size,
4 heads), `test_various_shapes_roundtrip`;
`tests/kv_block_roundtrip.rs` — `multi_scale_per_position_roundtrip`.

### Bug R5: `turbo_attention_forward` returned hardcoded statistics

**Severity**: HIGH — quality claims were fabricated.

The function returned `{snr_db: 20.0, cosine_similarity: 0.98,
max_abs_error: 0.01, mse: 0.0001}` regardless of input. It now computes
real statistics by comparing the fully-quantized attention path (Q
round-tripped through the quantizer, K/V from the compressed cache)
against the float-Q path written to `out`; end-to-end parity against
float attention is asserted in the test suite. The unused `mask` argument
is now honoured (additive score mask). Public signature unchanged.

**Tests**: `tests/attention_parity.rs` — output-vs-float-reference SNR
> 12 dB and cosine > 0.96, plus anti-hardcoding checks (stats must change
when inputs change and must not equal the old constants).

### Bug R6: 1-bit correction signs were silently dropped by the pipeline

**Severity**: MEDIUM — the QJL correction never survived storage.

`compress_tensor` computed the correction bits and threw them away
(`KvBlock` had no storage for them), so the decompressed tensors were
grid-only (~13 dB). `KvBlock` now has an optional `correction` field
(1 bit per value, lazily allocated); the memory/compression-ratio
accounting includes it honestly (default mode is ~3.8× vs FP16, not 5.3×).

**Tests**: `tests/attention_parity.rs`, `src/quantize.rs` round-trip tests.

### Bug R7: `turboquant-cpu` claimed SIMD it never used

The crate declared the `wide` SIMD crate (plus unused `bytemuck` and
`tracing`) and advertised "rayon + SIMD". No SIMD code existed. The unused
dependencies were removed and the docs now say rayon-only (scalar code is
auto-vectorized by LLVM). Unused `serde_json`/`bytemuck` were likewise
removed from `turboquant-core`.

### Bug R8: Documentation/config dishonesty

- `README.md`, `ci.yml`, `release.yml` referenced a nonexistent
  `--features cpu` cargo feature (the CPU backend is a crate, not a
  feature). Commands fixed to plain `--workspace` builds.
- `README.md` claimed 5.3×/6× compression "including scale overhead";
  the honest numbers are ~4.9× (no correction) and ~3.8× (default 1-bit
  correction). Table replaced with measured numbers.
- `pyproject.toml` declared the nonexistent build backend
  `setuptools.backends._legacy:_Backend`; fixed to
  `setuptools.build_meta`.
- This file's Bug→Test mapping referenced test files that did not exist;
  see the corrected table below.

---

## Bug→Test Mapping

Integration tests live in `rust/crates/turboquant-core/tests/`; unit
tests live in `#[cfg(test)]` modules inside the listed source files.

| Bug # | Test |
|-------|------|
| 1, 2 | `tests/bitpack_roundtrip.rs` (proptest round-trips) + `src/bitpack.rs` unit tests |
| 3 | `tests/kv_block_roundtrip.rs` — `partial_range_stores_do_not_overlap` |
| 4 | `tests/kv_block_roundtrip.rs` — `full_store_retrieve_roundtrip`, `multi_scale_per_position_roundtrip` |
| 5 | `src/kv_block.rs` unit tests (`test_new_block_sizes`, `test_store_and_retrieve`) |
| 6 | `src/qjl.rs` unit tests (`test_scale_adaptive`, `test_scale_fixed`); no dedicated optimality test yet |
| 7 | `tests/attention_parity.rs` |
| 8 | `src/qjl.rs` — `test_scale_percentile` (per-block percentile scaling) |
| 9 | `src/rotation.rs` — `test_qr_orthogonality`, `test_qr_preserves_norm` |
| 10 | `src/kv_block.rs` — `test_compression_ratio_includes_overhead` |
| 11 | `turboquant-bench` crate (criterion; see that crate) |
| 12 | calibration lives in `turboquant-cli`; see that crate's docs/tests |
| 13 | `turboquant-bench` crate (criterion; see that crate) |
| R1–R6 | listed per-bug above |
