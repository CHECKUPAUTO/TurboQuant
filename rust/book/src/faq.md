# FAQ

## Does TurboQuant work with any model?
Yes. It's model-agnostic — operates on KV cache tensors only.

## What's the quality impact?
Negligible. SNR > 12 dB on attention outputs.

## Can I use it with Ollama today?
Yes. Set cache-type turbo3 in your Modelfile.

## Does it require a GPU?
No. CPU backend achieves ~12 GB/s.
