# TurboQuant

**3-bit KV Cache Compression for Large Language Models**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A data-oblivious compression algorithm that reduces KV cache memory footprint by ~6x while maintaining near-zero quality loss (<0.1%).

## Overview

TurboQuant is a compression technique developed by Google Research (March 2026) that enables LLMs to operate with significantly larger context windows while reducing memory requirements.

### Key Benefits

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Bits per value | 16 (FP16) | 3 | ~6x reduction |
| Inference speed | Standard | Accelerated | Up to 8x |
| Quality loss | N/A | Quasi-zero | <0.1% |
| Context size | 2048-8192 | 16384+ | 2x+ |

## Algorithm

TurboQuant operates in two phases:

### Phase 1: PolarQuant (Geometric Rotation)

Applies a random orthogonal rotation matrix to distribute information uniformly:

```
y = R · x
```

Where:
- `x` is the original attention vector
- `R` is an orthogonal rotation matrix
- `y` is the rotated vector (easier to quantize)

This preserves ~99% of the useful signal while making quantization trivial.

### Phase 2: QJL (Quantized Johnson-Lindenstrauss)

Applies 1-bit residual correction to eliminate variance introduced by aggressive compression:

```
Correction = Sign(Residual) · Scale_factor
```

This ensures the dot product between Query and Key vectors remains mathematically exact.

## Installation

```bash
pip install -e .
# or
pip install torch  # minimal dependency
```

## Usage

### Python API

```python
from turboquant import TurboQuantKVCache, benchmark_turboquant

# Create compressed KV cache
cache = TurboQuantKVCache(
    num_layers=24,
    max_seq_len=4096,
    head_dim=128,
    num_heads=32,
    bits=3
)

# Compress during forward pass
k_compressed, k_scale = cache.compress(k_tensor, layer_idx=0)
v_compressed, v_scale = cache.compress(v_tensor, layer_idx=0)

# Decompress on retrieval
k_decompressed = cache.decompress(k_compressed, k_scale, layer_idx=0)

print(f"Memory: {cache.memory_usage_mb():.2f} MB")
print(f"Compression: {cache.compression_ratio_vs_fp16():.1f}x")
```

### Benchmark

```python
results = benchmark_turboquant(seq_len=4096)
print(f"FP16: {results['fp16_memory_mb']:.2f} MB")
print(f"TurboQuant: {results['turboquant_memory_mb']:.2f} MB")
print(f"Ratio: {results['compression_ratio']:.1f}x")
```

### llama.cpp Deployment

```bash
./llama-server -m /models/model_Q4_K_M.gguf \
    --port 11434 \
    --ctx-size 16384 \
    --cache-type-k turbo3 \
    --cache-type-v turbo3
```

### Ollama Modelfile (Future Native Support)

```dockerfile
# Modelfile for TurboQuant-enabled model
FROM llama3
PARAMETER kv_cache_type turbo3
PARAMETER num_ctx 16384
```

```bash
ollama create MyModel-Turbo -f Modelfile
```

## Verification

Check logs for successful activation:

```
[+] llama_new_context_with_model: kv self size = [Reduced Value]
[+] TURBOQUANT STATUS: ACTIVE (3-bit KV Cache)
[+] kv_cache_type_k: turbo3
[+] kv_cache_type_v: turbo3
```

## Mathematical Foundation

The algorithm leverages:

1. **Johnson-Lindenstrauss Lemma**: Dimensionality reduction preserving distances
2. **Orthogonal Matrices**: Rotation without distortion
3. **1-bit Quantization**: Minimal information loss with correction

### Compression Pipeline

```
Input Vector (FP16)
       │
       ▼
┌─────────────────┐
│  PolarQuant     │ ← Random orthogonal rotation
│  R · x          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3-bit Quantize │ ← Truncate to 3 bits
│  Q(y)           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QJL Correction │ ← 1-bit residual fix
│  Q(y) + ε       │
└────────┬────────┘
         │
         ▼
Compressed KV Cache (3-bit)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- For GPU: CUDA-capable device + `torch.cuda`
- Sufficient RAM for model weights (KV cache is compressed)

## Project Structure

```
TurboQuant/
├── turboquant.py         # Core implementation
├── README.md             # This file
├── LICENSE               # MIT License
├── requirements.txt      # Dependencies
├── pyproject.toml        # Package configuration
├── docs/
│   └── performance.md    # Benchmarks
└── MLA_TurboQuant_Synergy.md  # MLA integration
```

## Research Paper

Based on Google Research publication (March 2026). See `/docs` for technical details.

## License

MIT License - see [LICENSE](LICENSE) file.

## Author

CHECKUPAUTO — [GitHub](https://github.com/CHECKUPAUTO/TurboQuant)
