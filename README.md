# TurboQuant

**3-bit KV Cache Compression for Large Language Models**

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.83+-orange.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/license-PolyForm--NC%20%2B%20commercial-blue.svg)](LICENSING.md)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-red.svg)](https://blog.rust-lang.org/2024/11/28/Rust-1.83.0.html)
[![Book](https://img.shields.io/badge/book-mdBook-purple.svg)](https://checkupauto.github.io/TurboQuant)

A data-oblivious 3-bit compression algorithm that reduces KV cache memory
footprint by **~3.8–4.9×** (including scale/correction overhead) with a
measured round-trip SNR of ~13–19 dB on Gaussian data, depending on the
correction mode.

> ⚡ **Now in Rust** — see `rust/` for the maintained implementation.
> The Python prototype is preserved in `legacy/python/` for reference.

## Quickstart

```bash
# Install (Debian/Ubuntu/Fedora/Arch)
curl -fsSL https://raw.githubusercontent.com/CHECKUPAUTO/TurboQuant/main/rust/scripts/install.sh | bash

# Check it works
turboquant info

# Benchmark
turboquant bench --head-dim 128 --seq-len 4096

# Compress a GGUF model
turboquant compress model.gguf -o model-turbo3.gguf
```

## Measured Gains

| Mode | Bits/value | Ratio vs FP16 | Round-trip SNR (Gaussian) |
|------|------------|---------------|---------------------------|
| 3-bit, no correction | 3 + 0.25 (scales) | ~4.9× | ~13 dB |
| 3-bit + 1-bit residual correction (default) | 4 + 0.25 (scales) | ~3.8× | ~19 dB |

Ratios include per-block scale overhead (one f16 per 64 values) and, in
the default mode, the persisted 1-bit correction signs. SNR numbers are
measured by the test suite (`cargo test -p turboquant-core`) on
standard-normal data with block size 64.

## Algorithm

TurboQuant combines two techniques from the Johnson-Lindenstrauss literature:

### Phase 1: PolarQuant — Geometric Rotation

Apply a random orthogonal rotation matrix to distribute information uniformly:

```
y = x · R     where R is orthogonal (R · Rᵀ = I)
```

Available strategies: QR decomposition, Householder reflectors, Fast Hadamard Transform.

### Phase 2: QJL — Quantized Johnson-Lindenstrauss Correction

3-bit quantization with 1-bit residual sign correction:

```
q = Quantize3bit(x_norm) + Sign(residual) · ε
```

Guarantees: `|⟨Q(x), Q(y)⟩ - ⟨x, y⟩| < ε · ‖x‖ · ‖y‖`

## Architecture

```
rust/
├── crates/
│   ├── turboquant-core/     # Algorithms: rotation, quantize, bitpack
│   ├── turboquant-cpu/      # CPU backend (rayon + SIMD)
│   ├── turboquant-cuda/     # GPU backend (feature-gated)
│   ├── turboquant-gguf/     # GGUF I/O
│   ├── turboquant-cli/      # CLI binary
│   ├── turboquant-daemon/   # systemd service
│   ├── turboquant-ffi/      # C ABI (cbindgen)
│   ├── turboquant-py/       # Python bindings (pyo3)
│   └── turboquant-bench/    # Criterion benchmarks
├── scripts/                 # install.sh, update.sh, uninstall.sh
├── book/                    # mdBook documentation
└── docs/                    # Architecture, algorithm, benchmarks
```

## Documentation

- **[mdBook](https://checkupauto.github.io/TurboQuant)** — full user guide
- **[docs/ARCHITECTURE.md](rust/docs/ARCHITECTURE.md)** — internal design
- **[docs/ALGORITHM.md](rust/docs/ALGORITHM.md)** — mathematical derivation
- **[docs/USAGE.md](rust/docs/USAGE.md)** — recipes for llama.cpp, Ollama, C/FFI
- **[docs/MIGRATION_FROM_PYTHON.md](rust/docs/MIGRATION_FROM_PYTHON.md)** — porting guide

## Requirements

- Rust 1.83+ (MSRV)
- Debian/Ubuntu/Fedora/Arch x86_64
- Optional: CUDA 11.4+ for GPU backend
- For Python bindings: Python 3.10+, PyTorch 2.0+

## Building from Source

```bash
cd rust/
cargo build --release --workspace
cargo test --workspace
cargo doc --workspace --no-deps --open
```

(The CPU backend is the `turboquant-cpu` crate, built as part of the
workspace — there is no `cpu` cargo feature.)

## License

Double licence, alignée sur [SLHAv2](https://github.com/CHECKUPAUTO/SLHAv2) :
**PolyForm Noncommercial 1.0.0** (usage non-commercial et personnel, gratuit) +
**licence commerciale** requise pour tout usage commercial, offerte
exclusivement pour l'usage de TurboQuant comme module **CCOS** — voir
[LICENSING.md](LICENSING.md). TurboQuant et SLHAv2 sont des modules
compagnons de CCOS. Les versions antérieures publiées sous MIT/Apache-2.0
restent sous leurs termes d'origine.

## Author

CHECKUPAUTO — [GitHub](https://github.com/CHECKUPAUTO/TurboQuant)
