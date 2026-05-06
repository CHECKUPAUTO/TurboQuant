# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅                 |

## Reporting a Vulnerability

Please report security vulnerabilities privately via GitHub Security Advisories.
Do not open public issues for security bugs.

We will respond within 48 hours and provide a timeline for resolution.

## Security Design

- No `unsafe` without `// SAFETY:` justification
- Memory safety guaranteed by Rust's type system
- `ProtectSystem=strict` + `NoNewPrivileges=true` in systemd unit
- Daemon listens on `127.0.0.1` only (no external network exposure)
