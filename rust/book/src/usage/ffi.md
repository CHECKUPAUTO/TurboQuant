# C FFI

```c
#include "turboquant.h"
tq_kv_block_t* block = tq_compress(data, len, 3, 64);
tq_decompress(block, out, len);
tq_free_block(block);
```
