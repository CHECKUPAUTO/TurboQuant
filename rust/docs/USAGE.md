# Usage Guide

## CLI Recipes

### Compress a model for llama.cpp

```bash
turboquant compress llama-3-8b-Q4_K_M.gguf -o llama-3-8b-turbo3.gguf
```

### Verify compression integrity

```bash
turboquant verify llama-3-8b-turbo3.gguf
```

### Benchmark on your hardware

```bash
turboquant bench --head-dim 128 --seq-len 16384 --num-heads 32
```

### Audit Ollama models

```bash
turboquant audit ~/.ollama/models
```

### Start daemon

```bash
systemctl enable --now turboquant
turboquant daemon
```

## llama.cpp Integration

```bash
./llama-server -m model-turbo3.gguf \
    --port 11434 \
    --ctx-size 16384 \
    --cache-type-k turbo3 \
    --cache-type-v turbo3
```

## Ollama Modelfile

```dockerfile
FROM llama3
PARAMETER kv_cache_type turbo3
PARAMETER num_ctx 16384
```

```bash
ollama create MyModel-Turbo -f Modelfile
```

## Rust Library

```rust
use turboquant_core::rotation::QrRotation;
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};

let rot = QrRotation::new(128, Some(42));
let config = QjlConfig::default();
let quantizer = QjlQuantizer::new(config);
```

## Python Bindings

```python
import turboquant_py

cache = turboquant_py.TurboQuantKVCache(
    num_layers=24, max_seq_len=4096, head_dim=128, num_heads=32
)
print(cache.compression_ratio_vs_fp16())
```

## C/FFI

```c
#include "turboquant.h"

tq_kv_block_t* block = tq_compress(data, len, bits, block_size);
tq_decompress(block, out);
tq_free_block(block);
```
