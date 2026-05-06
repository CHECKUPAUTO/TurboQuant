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

## Use with llama.cpp

```bash
./llama-server -m my-model-turbo3.gguf --cache-type-k turbo3 --cache-type-v turbo3
```
