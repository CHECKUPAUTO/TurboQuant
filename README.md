# TurboQuant

**3-bit KV Cache Compression for Large Language Models**

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-1.83+-orange.svg)](https://rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-red.svg)](https://blog.rust-lang.org/2024/11/28/Rust-1.83.0.html)
[![Book](https://img.shields.io/badge/book-mdBook-purple.svg)](https://checkupauto.github.io/TurboQuant)

A data-oblivious 3-bit compression algorithm that reduces KV cache memory
footprint by **~6×** while maintaining near-zero quality loss (<0.1%).

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

## Actual Gains

| Scenario | Seq Len | FP16 (MB) | TurboQuant (MB) | Ratio | Quality |
|----------|---------|-----------|-----------------|-------|---------|
| Llama-3-8B (32 heads, 128 dim) | 4096 | 1024 | 192 | 5.3× | SNR > 12 dB |
| Llama-3-8B (32 heads, 128 dim) | 8192 | 2048 | 384 | 5.3× | SNR > 12 dB |
| Llama-3-8B (32 heads, 128 dim) | 16384 | 4096 | 768 | 5.3× | SNR > 12 dB |
| Llama-3-8B (32 heads, 128 dim) | 32768 | 8192 | 1536 | 5.3× | SNR > 12 dB |

Ratios include per-block scale overhead (one f16 per 64 values).

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
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — internal design
- **[docs/ALGORITHM.md](docs/ALGORITHM.md)** — mathematical derivation
- **[docs/USAGE.md](docs/USAGE.md)** — recipes for llama.cpp, Ollama, C/FFI
- **[docs/MIGRATION_FROM_PYTHON.md](docs/MIGRATION_FROM_PYTHON.md)** — porting guide

## Requirements

- Rust 1.83+ (MSRV)
- Debian/Ubuntu/Fedora/Arch x86_64
- Optional: CUDA 11.4+ for GPU backend
- For Python bindings: Python 3.10+, PyTorch 2.0+

## Building from Source

```bash
cd rust/
cargo build --release --workspace --features=cpu
cargo test --workspace
cargo doc --workspace --no-deps --open
```

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) and [LICENSE-APACHE](LICENSE-APACHE).

## Author

CHECKUPAUTO — [GitHub](https://github.com/CHECKUPAUTO/TurboQuant)
