# Usage Guide

## CLI Recipes

### Compress a GGUF model

```bash
turboquant compress llama-3-8b.gguf -o llama-3-8b-turbo3.gguf

# In place, custom block size / scale mode:
turboquant compress model.gguf --in-place --block-size 128 --scale-mode percentile
```

### Verify compression integrity

```bash
# Structure check: decompress every compressed tensor
turboquant verify llama-3-8b-turbo3.gguf

# Quality check: real per-tensor MSE/SNR against the uncompressed source
turboquant verify llama-3-8b-turbo3.gguf --original llama-3-8b.gguf
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

## llama.cpp / Ollama Integration

**Status: not supported by upstream llama.cpp or Ollama yet.** A turbo3
file is a valid GGUF container (see [GGUF.md](GGUF.md)), but stock
llama.cpp/Ollama will not reconstruct the compressed tensors — an engine
must implement the turbo3 spec (or link `turboquant-ffi`) to use them.
The invocation would look like:

```bash
# Hypothetical, requires a llama.cpp build that implements the turbo3 spec:
./llama-server -m model-turbo3.gguf \
    --ctx-size 16384 \
    --cache-type-k turbo3 \
    --cache-type-v turbo3
```

Until then, use `turboquant verify --original` to evaluate compression
quality offline.

## Rust Library

```rust
use turboquant_core::rotation::QrRotation;
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};

let rot = QrRotation::new(128, Some(42));
let config = QjlConfig::default();
let quantizer = QjlQuantizer::new(config);
```

## Python Bindings

Build with maturin (`maturin develop -m rust/crates/turboquant-py/Cargo.toml`);
the module name is `turboquant`:

```python
import numpy as np
import turboquant

data = (np.arange(64, dtype=np.float32) - 32.0) / 10.0

q = turboquant.Quantizer()  # 3-bit, absmax, 1-bit correction on
packed, scale, corr = q.quantize(data)
restored = q.dequantize(packed, len(data), scale, corr)
```

See [FFI.md](FFI.md) for the full Python API.

## C/FFI

```c
#include "turboquant.h"

tq_quantizer *q = NULL;
tq_quantizer_create(3, 64, TQ_SCALE_ABSMAX, 0.0f, 1, 0.0714f, &q);
tq_quantize(q, input, n, packed, tq_packed_size(n), &packed_len,
            &scale, corr, tq_corr_size(n), &corr_len);
tq_dequantize(q, packed, packed_len, n, scale, corr, corr_len, output, n);
tq_quantizer_destroy(q);
```

See [FFI.md](FFI.md) for the full C API and a complete example.
