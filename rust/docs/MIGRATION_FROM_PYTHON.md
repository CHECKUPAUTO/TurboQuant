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
7. **True 8-level grid and correction default** — the Rust quantizer uses
   the symmetric 3-bit grid `{-3.5, …, +3.5}` (spacing 1.0) with
   `DEFAULT_CORRECTION_SCALE = 0.25 / 3.5 ≈ 0.0714` relative to the block
   scale (quarter of the grid step). The Python prototype simulates
   quantization in floating point on a finer grid and uses a correction of
   `0.125` in its own level units — the constants are not interchangeable.

## Porting Checklist

- Replace `nn.Parameter(torch.zeros(..., dtype=torch.uint8))` with `KvBlock::new()`
- Replace `x.abs().max()` scaling with per-block `compute_scale()`
- Use `pack_3bit_slice` / `unpack_3bit_slice` for actual packing
- Instead of `learn_scale=True`, configure `CorrectionMode::OneBitResidual`
  (default `learned_scale` is the MSE-optimal quarter step; `turboquant
  calibrate` writes a starting-point YAML with these defaults — it does not
  yet learn from the data)
- Check quality with `turboquant verify <compressed> --original <source>`
  (upstream llama.cpp does not read turbo3 files yet; see GGUF.md)
