# Introduction

TurboQuant is a 3-bit KV cache compression algorithm for Large Language Models.

## What it does

- Compresses Key and Value caches from 16 bits per value (FP16) to ~3.25 bits
- Achieves ~5× memory reduction while keeping attention quality intact
- Enables 2-4× longer context windows on the same hardware

## How it works

1. **PolarQuant**: Apply random orthogonal rotation to spread information uniformly
2. **QJL**: 3-bit quantization with 1-bit residual correction
3. **Bit-packing**: Store 8 values in exactly 3 bytes

## Who should use it

- ML engineers running llama.cpp or Ollama with limited GPU memory
- Researchers exploring long-context LLM inference
- System integrators deploying models on edge hardware
