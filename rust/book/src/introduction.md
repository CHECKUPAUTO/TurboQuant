# Introduction

TurboQuant is a 3-bit KV cache compression algorithm for Large Language Models.

## What it does

- Compresses Key and Value caches from 16 bits per value (FP16) to
  3.25 bits (no correction) or 4.25 bits (default 1-bit correction),
  including per-block scale overhead
- Achieves ~3.8–4.9× memory reduction at ~13–19 dB round-trip SNR
  (measured on Gaussian data)
- Enables proportionally longer context windows on the same hardware

## How it works

1. **PolarQuant**: Apply random orthogonal rotation to spread information uniformly
2. **QJL**: 3-bit quantization with 1-bit residual correction
3. **Bit-packing**: Store 8 values in exactly 3 bytes

## Who should use it

- ML engineers running llama.cpp or Ollama with limited GPU memory
- Researchers exploring long-context LLM inference
- System integrators deploying models on edge hardware
