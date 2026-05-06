# Migration From Python

## API Correspondence

| Python (`turboquant.py`) | Rust (`turboquant-core`) |
|--------------------------|---------------------------|
| `PolarQuant(dim)` | `QrRotation::new(dim, seed)` |
| `pq.rotate(x)` | `rot.forward(&mut x)` |
| `QJLQuantizer(bits=3)` | `QjlQuantizer::new(QjlConfig::default())` |
| `quant.quantize(x)` | `quantizer.quantize_block(&block)` |
| `TurboQuantKVCache(...)` | `KvBlock::new(head_dim, seq_len, block_size)` |
| `cache.compress(k, 0)` | `compress_tensor(&polar, &quantizer, &k)` |
| `cache.compression_ratio_vs_fp16()` | `block.compression_ratio_vs_fp16()` |
| `benchmark_turboquant(...)` | `CpuBackend::new().full_benchmark(...)` |

## Key Differences

1. **No PyTorch dependency** — plain `Vec<u8>` + `Vec<f16>`
2. **True 3-bit packing** — not stubbed 8-bit
3. **Per-block scaling** — not global `abs().max()`
4. **Decompressed attention** — forward actually uses compressed cache
5. **Scale overhead** — included in ratio calculation
6. **Multiple rotation strategies** — QR, Householder, Hadamard

## Porting Checklist

- Replace `nn.Parameter(torch.zeros(..., dtype=torch.uint8))` with `KvBlock::new()`
- Replace `x.abs().max()` scaling with per-block `compute_scale()`
- Use `pack_3bit_slice` / `unpack_3bit_slice` for actual packing
- Run `turboquant calibrate` instead of `learn_scale=True`
- Set `--cache-type-k turbo3` in llama.cpp server arguments
