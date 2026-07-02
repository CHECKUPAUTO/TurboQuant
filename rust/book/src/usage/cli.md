# CLI Usage

```bash
# Compress a GGUF model (writes <name>-turbo3.gguf unless -o/--output is given)
turboquant compress model.gguf -o model-turbo3.gguf
turboquant compress model.gguf --in-place --block-size 128 --scale-mode percentile

# Verify: decompress every compressed tensor; with --original, report
# real per-tensor MSE/SNR against the uncompressed source
turboquant verify model-turbo3.gguf
turboquant verify model-turbo3.gguf --original model.gguf

# Benchmark compression on this machine
turboquant bench --head-dim 128 --seq-len 16384 --num-heads 32

# Write a starting-point calibration YAML (default parameters)
turboquant calibrate calibration.npy -o calibration.yaml

# Estimate savings for a folder of GGUF models
turboquant audit ~/.ollama/models

# System info: version, backends, CPU threads
turboquant info

# Run the watch-and-compress daemon (usually via systemd)
turboquant daemon
turboquant daemon --config /etc/turboquant/config.json
```

Compress options: `--output/-o` (single file), `--in-place`
(mutually exclusive with `--output`), `--bits` (only 3 is implemented),
`--block-size` (power of two ≥ 8, default 64), `--scale-mode`
(`absmax`, `percentile`, `adaptive`).

Global options: `-v/-vv/-vvv` (verbosity), `-j/--threads`, `--backend`
(`cpu` or `auto`; there is no CUDA backend), `--color`.
