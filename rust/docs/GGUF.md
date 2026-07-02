# GGUF Format — TurboQuant Extensions

This document describes (1) the subset of the GGUF container format
implemented by the `turboquant-gguf` crate, and (2) the TurboQuant
"turbo3" compressed-model format (version 1) that `turboquant compress`
writes on top of it.

## 1. GGUF container support

`turboquant-gguf` reads GGUF **v2 and v3** and writes **v3**
(little-endian throughout):

```
header:       magic "GGUF" (4 bytes)
              version           u32   (3 on write)
              tensor_count      u64
              metadata_kv_count u64
metadata:     metadata_kv_count × (key: string, value_type: u32, value)
tensor infos: tensor_count × (name: string, n_dims: u32,
              dims[n_dims]: u64, ggml_type: u32, offset: u64)
padding:      zeros up to the next multiple of ALIGNMENT
tensor data:  each tensor at data_start + offset,
              offsets aligned to ALIGNMENT
```

* Strings are `u64` length + UTF-8 bytes (no terminator).
* All 13 metadata value types are supported: `u8,i8,u16,i16,u32,i32,f32,
  bool,string,array,u64,i64,f64`. Arrays are homogeneous
  (`elem_type u32 + count u64 + elements`) and may nest.
* `ALIGNMENT` comes from the `general.alignment` metadata key
  (power of two, default **32**). The writer emits this key
  automatically when a non-default alignment is set.
* Tensor `offset`s are relative to the start of the (aligned) data
  section, and each is itself aligned.
* Tensor data access: `F32` and `F16` decode to `f32`; every other ggml
  type is exposed as raw bytes (exact length for fixed-size element
  types, the span up to the next tensor offset otherwise).

Rust API: `GgufParser::parse` / `GgufParser::parse_file` → `GgufFile`
(header, metadata, tensor infos, tensor data), and `GgufWriter`
(`add_metadata`, `add_tensor`, `to_bytes`, `write_to_file`).
Round-trip (`parse(write(x)) == x`) is covered by tests in
`crates/turboquant-gguf/tests/roundtrip.rs`.

## 2. TurboQuant "turbo3" format, version 1

A compressed model is a **standard GGUF v3 file**. A plain GGUF reader
can still open it; float tensors are simply replaced by flat byte
tensors plus `turboquant.*` metadata that describes how to reconstruct
them. Implemented in `turboquant_gguf::turbo`.

### Global metadata keys

| Key | Type | Description |
|-----|------|-------------|
| `turboquant.format_version` | UINT32 | Format version (**1**). Presence marks a file as TurboQuant-compressed. |
| `turboquant.version` | STRING | Version of the tool that wrote the file. |
| `turboquant.bits` | UINT8 | Quantization bits (only **3** is implemented). |
| `turboquant.block_size` | UINT32 | Values per scale block (power of two ≥ 8, default 64). |
| `turboquant.scale_mode` | STRING | `absmax`, `percentile:<p>`, `adaptive`, or `fixed:<s>` (informational). |
| `turboquant.correction` | STRING | `none` or `one_bit_residual`. |
| `turboquant.correction_scale` | FLOAT32 | Learned scale of the 1-bit residual correction (present iff `one_bit_residual`). |

All metadata of the source model is carried through unchanged.

### Per-tensor metadata keys

For every compressed tensor `<name>`:

| Key | Type | Description |
|-----|------|-------------|
| `turboquant.<name>.orig_type` | STRING | Original element type: `F32` or `F16`. |
| `turboquant.<name>.orig_shape` | ARRAY of UINT64 | Original dims (ggml order, innermost first). |
| `turboquant.<name>.scales` | ARRAY of FLOAT32 | One scale per block, `ceil(n / block_size)` entries, in block order. Values are f16-exact. |

Tensors **without** these keys (all non-F32/F16 tensors) are passed
through byte-for-byte with their original type and dims.

### Compressed tensor payload

Each compressed tensor is stored under its **original name** as a 1-D
`GGML_TYPE_I8` tensor whose single dim is the payload byte length:

```
[ packed 3-bit codes, block by block ][ correction bits, block by block ]
```

With `n` original elements and `B = block_size`, block `i` holds
`n_i = min(B, n - i·B)` values:

* **Packed section** — per block: `n_i` 3-bit code indices (0..7),
  zero-padded to a multiple of 8 values, packed 8-values-into-3-bytes
  (`turboquant_core::bitpack`), i.e. `ceil8(n_i)·3/8` bytes.
* **Correction section** — present iff `turboquant.correction` is
  `one_bit_residual`; per block: `ceil(n_i/8)` bytes of LSB-first
  residual sign bits.

Reconstruction per value (see `turboquant_core::qjl`): the 3-bit index
maps back to a level in `[-1, 1]` scaled by the block scale, plus
`±correction_scale × scale` from the correction bit when enabled.
Cost: 3 bits/value + 1 correction bit/value + one f16-precision scale
per block (stored as f32 metadata) — ~4.5 bits/value vs 32 for F32.

### Rules

* Compressing an already-compressed file (detected via
  `turboquant.format_version`) is refused.
* Decompression: `turbo::decompress_tensor(&file, name) -> Vec<f32>`;
  `turboquant verify <file> --original <source>` reports real per-tensor
  MSE/SNR.
* Standard GGUF readers see a valid file with flat `I8` tensors; they
  will not reconstruct floats without implementing this spec.
