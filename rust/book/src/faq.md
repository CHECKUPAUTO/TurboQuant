# FAQ

## Does TurboQuant work with any model?
Yes. It's model-agnostic — it operates on tensors/KV cache values only.

## What's the quality impact?
Measured on Gaussian data: round-trip SNR ~19 dB with the default 1-bit
correction (~13 dB without); attention-output SNR > 12 dB and cosine
similarity > 0.96 in the test suite. No model-level perplexity results yet.

## Can I use it with llama.cpp or Ollama today?
Not directly. `turboquant compress` writes valid GGUF files ("turbo3"
format), but upstream llama.cpp/Ollama do not implement the turbo3 spec
yet, so they cannot reconstruct the compressed tensors. Use
`turboquant verify --original` to evaluate quality offline.

## Does it require a GPU?
No. Everything runs on the CPU (rayon-parallel). There is no CUDA
backend yet — `turboquant-cuda` is a placeholder crate.
