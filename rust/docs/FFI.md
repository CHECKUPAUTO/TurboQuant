# FFI: C ABI and Python bindings

TurboQuant ships two binding layers over `turboquant-core`:

- **`turboquant-ffi`** — a C ABI (`cdylib` + `staticlib`) with the header
  `crates/turboquant-ffi/include/turboquant.h`.
- **`turboquant-py`** — a Python extension module `turboquant` built with
  pyo3 (+ maturin).

---

## C API (`turboquant-ffi`)

### Header and libraries

```c
#include "turboquant.h"
```

The header lives at `crates/turboquant-ffi/include/turboquant.h` and is
regenerated from the Rust source by cbindgen at build time (`build.rs`), so
it always matches the compiled ABI. Build produces both
`libturboquant_ffi.a` (static) and `libturboquant_ffi.so` (shared):

```sh
cargo build -p turboquant-ffi --release
cc app.c -I crates/turboquant-ffi/include \
   -L target/release -lturboquant_ffi -lpthread -ldl -lm
```

### Design rules

- Every fallible function returns an `int` status code; `TQ_OK` (0) means
  success.
- Every pointer argument is validated; NULL never crashes, it returns
  `TQ_ERR_NULL_POINTER`.
- No panics cross the FFI boundary: inputs are validated up front and core
  calls are additionally wrapped in `catch_unwind`.
- No global state. The only stateful object is the opaque `tq_quantizer*`
  handle, which is immutable after creation and safe to share across
  threads.
- Callers own all buffers; the library never allocates memory the caller
  must free (except the handle, released with `tq_quantizer_destroy`).

### Status codes

| Code | Value | Meaning |
|---|---|---|
| `TQ_OK` | 0 | Success |
| `TQ_ERR_NULL_POINTER` | 1 | A required pointer argument was NULL |
| `TQ_ERR_INVALID_ARGUMENT` | 2 | Bad enum value, out-of-range parameter, `n == 0`, non-finite float, unsupported `bits`, ... |
| `TQ_ERR_BUFFER_TOO_SMALL` | 3 | An output/input buffer is smaller than the documented minimum |
| `TQ_ERR_INTERNAL` | 4 | Internal error (should not occur for valid inputs) |

### Scale modes

| Constant | Value | Scale is computed as | `scale_param` |
|---|---|---|---|
| `TQ_SCALE_ABSMAX` | 0 | max(&#124;x&#124;) of the block | ignored |
| `TQ_SCALE_PERCENTILE` | 1 | given percentile of &#124;x&#124; | percentile in [0, 1] |
| `TQ_SCALE_ADAPTIVE` | 2 | standard deviation of the block | ignored |
| `TQ_SCALE_FIXED` | 3 | `scale_param` itself | finite, > 0 |

### Functions

```c
// Version string ("1.0.0-rc1"); static, do not free.
const char *tq_version(void);

// Buffer sizing. Packed output: n padded to a multiple of 8 values,
// 3 bits/value => tq_packed_size(64) == 24. Correction output: 1 bit/value
// => tq_corr_size(64) == 8. Both return 0 for n == 0.
size_t tq_packed_size(size_t n);
size_t tq_corr_size(size_t n);

// Create / destroy the opaque quantizer handle.
// bits must be 3. block_size (> 0) is nominal: each tq_quantize call
// quantizes its whole input as ONE block with ONE scale, so callers chunk
// their data (typically block_size values per call).
// correction_enabled != 0 turns on 1-bit residual correction with magnitude
// correction_scale (relative to the block scale; 0.0714f = quarter step
// is the MSE-optimal fixed value, cf. DEFAULT_CORRECTION_SCALE in core).
int tq_quantizer_create(uint8_t bits,
                        size_t block_size,
                        int scale_mode,
                        float scale_param,
                        int correction_enabled,
                        float correction_scale,
                        tq_quantizer **out_quantizer);
void tq_quantizer_destroy(tq_quantizer *quantizer);   // NULL is a no-op

// Quantize n floats (n > 0) as a single block.
//   packed_out   : capacity packed_cap >= tq_packed_size(n)
//   packed_written: receives bytes written
//   scale_out    : receives the block scale (round-trips through f16)
//   corr_out     : required with capacity corr_cap >= tq_corr_size(n) when
//                  correction is enabled; may be NULL otherwise
//   corr_written : optional (may be NULL); 0 when correction is disabled
int tq_quantize(const tq_quantizer *quantizer,
                const float *input, size_t n,
                uint8_t *packed_out, size_t packed_cap, size_t *packed_written,
                float *scale_out,
                uint8_t *corr_out, size_t corr_cap, size_t *corr_written);

// Dequantize back to n floats.
//   packed_len >= tq_packed_size(n); output_cap >= n floats
//   scale: the value returned by tq_quantize (finite, > 0)
//   corr: optional; when non-NULL, corr_len >= tq_corr_size(n). Passing
//   NULL skips the residual correction.
int tq_dequantize(const tq_quantizer *quantizer,
                  const uint8_t *packed, size_t packed_len,
                  size_t n, float scale,
                  const uint8_t *corr, size_t corr_len,
                  float *output, size_t output_cap);
```

### Example

```c
#include <stdio.h>
#include <stdlib.h>
#include "turboquant.h"

int main(void) {
    tq_quantizer *q = NULL;
    if (tq_quantizer_create(3, 64, TQ_SCALE_ABSMAX, 0.0f, 1, 0.0714f, &q) != TQ_OK)
        return 1;

    enum { N = 64 };
    float input[N], output[N];
    for (int i = 0; i < N; i++) input[i] = (float)(i - 32) / 10.0f;

    uint8_t *packed = malloc(tq_packed_size(N));
    uint8_t *corr   = malloc(tq_corr_size(N));
    size_t packed_len = 0, corr_len = 0;
    float scale = 0.0f;

    if (tq_quantize(q, input, N, packed, tq_packed_size(N), &packed_len,
                    &scale, corr, tq_corr_size(N), &corr_len) != TQ_OK)
        return 2;
    if (tq_dequantize(q, packed, packed_len, N, scale,
                      corr, corr_len, output, N) != TQ_OK)
        return 3;

    printf("scale=%f first=%f\n", scale, output[0]);
    free(packed); free(corr);
    tq_quantizer_destroy(q);
    return 0;
}
```

A full smoke test that compiles and runs a C program like this lives in
`crates/turboquant-ffi/tests/c_smoke.rs` (skipped when no `cc` is on PATH).

---

## Python API (`turboquant-py`)

Build with maturin (module name is `turboquant`):

```sh
pip install maturin numpy
maturin develop -m rust/crates/turboquant-py/Cargo.toml --release
```

### Surface

```python
import numpy as np
import turboquant

turboquant.__version__                     # e.g. "1.0.0-rc1"

q = turboquant.Quantizer(
    bits=3,                # only 3 is supported
    block_size=64,         # nominal; each quantize() call is one block
    scale_mode="absmax",   # 'absmax' | 'percentile' | 'adaptive' | 'fixed'
    percentile=None,       # required for scale_mode='percentile', in [0, 1]
    correction=True,       # 1-bit residual correction
    correction_scale=None, # default 0.25/3.5 ≈ 0.0714 (quarter step), relative to the block scale
    fixed_scale=None,      # required for scale_mode='fixed', > 0
)
q.bits, q.block_size, q.scale_mode, q.correction   # read-only attributes

# quantize: 1-D float32 array -> (packed, scale, correction)
#   packed:     np.uint8, 3 * ceil(n / 8) bytes
#   scale:      float (round-trips through f16)
#   correction: np.uint8, ceil(n / 8) bytes, or None if correction=False
packed, scale, correction = q.quantize(data)

# dequantize: back to n float32 values; correction=None skips correction
restored = q.dequantize(packed, n, scale, correction)

# 3-bit packing helpers (values in 0..=7, length a multiple of 8)
packed_bits = turboquant.pack_3bit(values)          # uint8 -> uint8
values      = turboquant.unpack_3bit(packed_bits, n)

# Seeded fast Hadamard rotation (length must be a power of two)
y = turboquant.hadamard_rotate(x, seed=42)
x = turboquant.hadamard_rotate(y, seed=42, inverse=True)
```

All argument errors (wrong bits, unknown scale mode, missing/out-of-range
parameters, empty input, too-small buffers, non-power-of-two lengths,
non-contiguous arrays) raise `ValueError`.

### Example

```python
import numpy as np
import turboquant

data = (np.arange(64, dtype=np.float32) - 32.0) / 10.0

q = turboquant.Quantizer()                 # 3-bit, absmax, correction on
packed, scale, corr = q.quantize(data)
restored = q.dequantize(packed, len(data), scale, corr)

mse = float(np.mean((data - restored) ** 2))
print(f"scale={scale:.4f} mse={mse:.6f}")
```

A runnable smoke script lives at `crates/turboquant-py/tests/smoke.py`
(not part of CI; requires numpy and a `maturin develop` build).

### Notes

- Inputs must be C-contiguous 1-D arrays (`np.ascontiguousarray` if
  needed); outputs are freshly allocated arrays moved out of Rust without
  an extra copy.
- `cargo test -p turboquant-py` unit-tests the pure-Rust layer behind the
  bindings; building the importable module requires maturin because pyo3
  is compiled with the `extension-module` feature.
