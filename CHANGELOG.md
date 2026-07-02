# Changelog

All notable changes to TurboQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-07-02

### Changed
- The attached `LICENSE.md` carries the **canonical** PolyForm
  Noncommercial 1.0.0 text (15 sections, from polyformproject.org) with the
  Required Notice — not a paraphrase. The CCOS exclusivity applies to the
  commercial offer; noncommercial use is standard, unrestricted PolyForm NC.
- **License**: relicensed from MIT OR Apache-2.0 to a dual license —
  PolyForm Noncommercial 1.0.0 (free for noncommercial use) plus a
  commercial license offered exclusively for use as a CCOS module
  (TurboQuant and SLHAv2 are companion modules of CCOS). Prior published
  versions remain available under their original MIT OR Apache-2.0 terms.
  See LICENSING.md

### Fixed
- **QJL 3-bit grid collapse** (`turboquant-core`): the quantizer mapped
  values onto a mis-sized 15-level half-step grid (increment 0.5) and
  clamped the index to 8 codes, so every positive value decoded to 0.
  Replaced with the symmetric 8-level grid `{-3.5 … +3.5}` (spacing 1.0,
  `idx = clamp(round(x_norm·3.5 + 3.5), 0, 7)`). Measured round-trip SNR
  on Gaussian data: ~13 dB without correction, ~19 dB with it (was ~3 dB)
- 1-bit residual sign computed in consistent level units (was mixed
  normalized/level units, pushing corrections the wrong way)
- 1-bit correction default raised from the harmful `0.01` to the
  MSE-optimal quarter step (`DEFAULT_CORRECTION_SCALE = 0.25/3.5 ≈ 0.0714`)
- Multi-block position collisions in `compress_tensor`/`KvBlock` (silent
  data corruption for `head_dim > block_size`)
- `turbo_attention_forward` returned hardcoded statistics; it now computes
  real SNR/cosine/MSE and honours the attention mask
- 1-bit correction signs are now persisted in `KvBlock` and the GGUF
  payload (previously computed then discarded); memory accounting includes
  them (default mode is ~3.8× vs FP16, not 5.3×)
- Python prototype (`turboquant.py`): QJL correction was scaled by the
  input's dynamic range (`× x_max`); it now stays in level units with the
  MSE-optimal default of 0.125
- CI: repaired the never-green mdBook, Security Audit, and Python jobs;
  removed the nonexistent `--features cpu` flag from workflows
- Documentation aligned with measured reality (compression ratios, SNR,
  no fake llama.cpp/Ollama support claims, honest CUDA stub)

### Added
- Real GGUF v2/v3 parsing and v3 writing, plus the "turbo3" compressed
  model format v1 (`turboquant.*` metadata; documented in rust/docs/GGUF.md)
- Working CLI: `compress` (`--output`/`--in-place`/`--bits`/`--block-size`/
  `--scale-mode`), `verify` (real per-tensor SNR/MSE with `--original`),
  `bench`, `calibrate`, `audit`, `info`, `daemon`
- Functional daemon: filesystem watcher that auto-compresses `.gguf`
  files, `GET /healthz` endpoint, systemd `Type=notify` readiness
- C ABI (`turboquant-ffi`): `tq_quantizer_create/destroy`, `tq_quantize`,
  `tq_dequantize`, buffer-sizing helpers; cbindgen header + C smoke test
  (was an empty stub)
- Python bindings (`turboquant-py`, module `turboquant`): `Quantizer`,
  `pack_3bit`/`unpack_3bit`, `hadamard_rotate` (was an empty stub)
- Regression tests for every fixed bug (grid round-trip signs, correction
  SNR gain, dot-product preservation, attention parity anti-hardcoding)

### Security
- pyo3 upgraded 0.22.6 → 0.29.0; `anyhow` and `memmap2` bumped to patched
  versions

## [1.0.0-rc1] — 2026-05-06

### Added
- Full Rust workspace with 9 crates
- PolarQuant rotation: QR, Householder, Fast Hadamard
- QJL 3-bit quantizer with adaptive scale mode
- True 3-bit bit-packing (8 values → 3 bytes)
- CPU backend with rayon parallelism and SIMD
- CLI with compress, verify, bench, calibrate, audit, info, daemon
- GGUF I/O framework
- systemd daemon with HTTP API
- C FFI bindings
- Python bindings via pyo3
- Shell install/update/uninstall scripts
- Comprehensive mdBook documentation
- GitHub Actions CI/CD

### Fixed
- 13 bugs documented in BUGS_FIXED.md
- True bit-packing (was stubbed as 8-bit)
- Per-block scaling (was global)
- Attention forward now uses decompressed K/V
- Scale overhead included in compression ratio
- Orthogonal matrix convention documented
- All `unwrap`/`expect`/`panic` removed from production code

### Changed
- Migrated from single-file Python to multi-crate Rust workspace
- Python code preserved in `legacy/python/`

### Deprecated
- Python `turboquant.py` — use Rust crate instead
