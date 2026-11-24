# ffpix

Memory-safe, zero-dependency, demand-driven streaming image processing engine,
C99 ABI library, WebAssembly module, and standalone CLI written in stable Rust
(Edition 2024).

## Engineering Guarantees

- **Zero External Dependencies:** Every workspace crate uses only Rust `std`
  (`core`/`alloc`). No third-party crates are linked.
- **Strict Memory Safety:** `#![forbid(unsafe_code)]` is enforced at the root
  of all core, DAG, color, metadata, codec, operator, vector, CLI, and
  verification crates. `unsafe` is isolated exclusively to FFI boundaries and
  runtime-dispatched SIMD hot loops backed by bit-exact scalar references.
- **Zero Panics on Untrusted Input:** All dimension, stride, offset, and buffer
  calculations use checked arithmetic and return deterministic `FfpError` codes.
- **Hard Resource Budgets:** Decoding, tile allocation, metadata parsing, and
  decompression flow through an explicit `Budget` enforcing dimension, memory,
  frame, metadata, recursion, and decompression-ratio caps.
- **Cross-Platform Pure Computation:** No OS windowing, GUI, or system graphics
  library dependencies.

## Workspace Layout

Crates are built incrementally according to `PLAN.md`:

- `crates/ffpix-core` - Core types, checked geometry, pixel layouts, `Budget`,
  tile buffers, streaming I/O traits, and deterministic error codes.

## Building and Testing

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Documentation

- `IDEA.md` - Master engineering specification and subsystem architecture.
- `PLAN.md` - Granular 240-commit implementation plan and verification gates.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
