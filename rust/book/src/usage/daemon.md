# Daemon

`turboquant daemon` (or the `turboquant-daemon` binary) watches
configured directories for new `.gguf` files and compresses each into an
output directory. It integrates with systemd (`Type=notify` readiness)
and shuts down gracefully on SIGTERM/ctrl-c.

## systemd watchdog

When the unit sets `WatchdogSec=` (the shipped unit uses 60 seconds),
systemd exports `WATCHDOG_USEC` (and `WATCHDOG_PID`) to the daemon,
which then sends `WATCHDOG=1` keepalives at half that interval (clamped
to at least 1 second). If the daemon's event loop hangs, the keepalives
stop and systemd restarts the service. `WATCHDOG_PID` is honored: if it
names another process, no keepalives are sent. Outside systemd (no
watchdog environment), no keepalive task runs at all.

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
