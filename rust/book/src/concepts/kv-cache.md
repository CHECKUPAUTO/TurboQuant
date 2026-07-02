# KV Cache

The KV cache is the memory bottleneck in transformer inference.
During autoregressive generation, every attention layer stores intermediate
K and V tensors for all previous tokens.

## Memory Footprint

```text
KV_memory = 2 × num_layers × seq_len × num_heads × head_dim × 2 bytes
```

For Llama-3-8B at seq_len=4096: ~1 GB per request.

## Why Compress?

- Longer contexts need proportional memory
- Batched inference multiplies memory linearly
- Consumer GPUs (8-12 GB VRAM) hit limits quickly
