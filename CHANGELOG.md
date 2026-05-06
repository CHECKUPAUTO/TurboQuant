# Changelog

All notable changes to TurboQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
