# Flox packaging notes

The package expression is intentionally small:

- `src = ../../..;` points from `.flox/pkgs/ollama-model-resolver/default.nix` back to the repository root.
- `rustPlatform.buildRustPackage rec { ... }` allows `cargoLock.lockFile = "${src}/Cargo.lock"` to refer to the source path.
- `doCheck = true` runs the Rust test suite during the package build.
- `reqwest` uses `rustls-tls`, so the package does not need OpenSSL build inputs.

Build locally with:

```bash
flox build ollama-model-resolver
```
