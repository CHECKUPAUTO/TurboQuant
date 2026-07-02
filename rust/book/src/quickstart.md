# Quickstart

## Check your setup

```bash
turboquant info
```

## Run a benchmark

```bash
turboquant bench --head-dim 128 --seq-len 4096
```

## Compress a model

```bash
turboquant compress my-model.gguf -o my-model-turbo3.gguf
```

## Verify the result

```bash
# Structural check
turboquant verify my-model-turbo3.gguf

# Real per-tensor SNR/MSE against the original
turboquant verify my-model-turbo3.gguf --original my-model.gguf
```

Note: turbo3 files are valid GGUF containers, but upstream llama.cpp and
Ollama do not implement the turbo3 spec yet, so they cannot use the
compressed tensors directly. See the [CLI](usage/cli.md) page for all
commands.
