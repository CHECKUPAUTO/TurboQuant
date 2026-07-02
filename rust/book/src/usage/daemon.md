# Daemon

`turboquant daemon` (or the `turboquant-daemon` binary) watches
configured directories for new `.gguf` files and compresses each into an
output directory. It integrates with systemd (`Type=notify` readiness)
and shuts down gracefully on SIGTERM/ctrl-c.

## Configuration

JSON config file (`turboquant daemon --config config.json`); defaults:

- `watch_dirs`: directories scanned for `.gguf` files
- `output_dir`: `~/.turboquant/compressed`
- `listen_addr`: `127.0.0.1:7460`
- `block_size`, `interval_secs` (debounce)

## HTTP API

A single health endpoint:

```text
GET /healthz   ->  {"status":"ok","files_compressed":N,"failures":M}
```
