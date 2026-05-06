# Library Usage

```rust
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};

let config = QjlConfig::default();
let quantizer = QjlQuantizer::new(config);
let compressed = quantizer.quantize_block(&data);
```
