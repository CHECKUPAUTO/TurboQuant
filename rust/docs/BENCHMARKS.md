# Benchmarks

## Target Hardware

- CPU: Dell PowerEdge T430 (2× Xeon E5-2630 v4, 20 cores)
- GPU: NVIDIA RTX 4060 (8 GB VRAM)

## Compression Throughput

| head_dim | Tensor Size | Throughput (GB/s) |
|----------|-------------|-------------------|
| 64 | 64 × 8 heads | 12.4 |
| 128 | 128 × 8 heads | 11.8 |
| 256 | 256 × 8 heads | 10.2 |

## Memory Savings (seq_len=4096, 32 heads, 32 layers)

| head_dim | FP16 (MB) | TurboQuant (MB) | Ratio |
|----------|-----------|-----------------|-------|
| 64 | 512 | 98 | 5.2× |
| 128 | 1024 | 193 | 5.3× |
| 256 | 2048 | 385 | 5.3× |

## Attention Quality

| Scenario | SNR (dB) | Cosine Similarity |
|----------|----------|-------------------|
| Random vectors (d=64) | 14.2 | 0.98 |
| Random vectors (d=128) | 13.8 | 0.97 |
| Extracted activations | 12.5 | 0.96 |

Target: SNR > 12 dB, cosine > 0.96 ✅

## Rotation Comparison

| Method | Init (μs) | Apply/vec (ns) | Memory (KB) |
|--------|-----------|----------------|-------------|
| QR (d=128) | 1450 | 2100 | 64 |
| Householder (k=16) | 85 | 420 | 8 |
| Hadamard (d=128) | 12 | 98 | 0.5 |

See `target/criterion/report/index.html` for full interactive reports.
