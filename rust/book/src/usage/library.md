# Library Usage

```rust
extern crate turboquant_core;
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};

fn main() {
    let config = QjlConfig::default();
    let quantizer = QjlQuantizer::new(config);

    // One block of values to compress (any &[f32]).
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();

    let compressed = quantizer.quantize_block(&data);
    let restored = quantizer.dequantize_block(&compressed, data.len());

    assert_eq!(restored.len(), data.len());
    println!("scale = {}", compressed.scale);
}
```

This example is compiled and run by `mdbook test` in CI against the real
`turboquant-core` crate, so it cannot drift from the actual API.

See the [C FFI](ffi.md) and [CLI](cli.md) pages for the other entry points.
