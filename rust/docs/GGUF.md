# GGUF Format — TurboQuant Extensions

## New Metadata Keys

| Key | Type | Description |
|-----|------|-------------|
| `turboquant.cache_type_k` | STRING | `"turbo3"` for 3-bit K cache |
| `turboquant.cache_type_v` | STRING | `"turbo3"` for 3-bit V cache |
| `turboquant.bits` | UINT8 | Quantization bits (default: 3) |
| `turboquant.block_size` | UINT32 | Per-block scale granularity |
| `turboquant.format_version` | UINT32 | Binary format version (1) |

## Block Layout

When `cache-type` is `turbo3`, KV cache tensors are stored as:

```
[packed_3bit_data] [per_block_scales (f16)] [correction_bits (optional)]
```

Backward compatible: standard GGUF readers see a flat uint8 tensor.
