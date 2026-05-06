# CLI Usage

```bash
turboquant compress model.gguf -o model-turbo3.gguf
turboquant verify model-turbo3.gguf
turboquant bench --head-dim 128 --seq-len 16384
turboquant calibrate calibration.npy
turboquant audit ~/.ollama/models
turboquant info
turboquant daemon
```
