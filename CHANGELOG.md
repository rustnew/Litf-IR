# Changelog

All notable changes to LIFT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (v0.5)
- State-vector quantum simulator (CPU, up to ~25 qubits)
- Tensor interpreter (numpy-like execution of tensor ops)
- Real LLVM IR lowering with cuBLAS/cuDNN runtime calls
- Functional importers — ONNX, PyTorch FX, OpenQASM 3 (currently stubs)
- SABRE-style dynamic qubit re-placement

### Planned (v0.6)
- True automatic differentiation (backward graph construction)
- PyO3 Python bindings
- Multi-file support (`include` / linking)
- v1.0 release — full pipeline, benchmarks, arXiv paper

## [0.4.5] — 2026-08-25

### Fixed
- **Printer/parser round trip** — `optimise --output out.lif` produced a
  `.lif` file the parser could not read back (the printer emitted disconnected
  signature names plus a `^bb0(...):` block label with no grammar rule for
  it). The signature now prints the entry block's real argument names, and
  the redundant block header is no longer emitted.
- **QASM export qubit indexing** — gates were numbered from a running counter
  instead of their actual operand, so any two gates in a row could land on
  different qubits and `CX`'s control/target could come out swapped. Qubit
  indices are now resolved by walking each operand's SSA def chain back to
  its owning qubit. Also fixed gate export order (was iterating the ops
  slotmap, which drifts once a pass frees a slot and a later pass reuses it;
  now walks `block.ops` in program order) and per-function qubit counting
  (was summing qubit-typed block args across every function in the module).
- **`gate-decomposition` no longer doubles the transformation** — the pass
  built a native decomposition chain but left the original gate in the block,
  still wired to produce its own result, so e.g. decomposing `T` silently
  produced `Rz(pi/4)` *followed by* the still-present `T` (i.e. `S`, not
  `T`'s actual decomposition). The original gate's results are now redirected
  to the decomposition chain's output and the original op is removed.
- **RX decomposition sign error** — the first `Rz` in the native `RX(theta)`
  sequence had the wrong sign, so `RX(0)` compiled to `Z` instead of the
  identity, for every angle. Contributed by @cleitonaugusto (#4), verified
  independently against the closed-form `RX(theta)` matrix at 8 angles.
- CLI `--version` was hardcoded to `"0.3.0"` from an earlier release; now
  reads the real crate version via `CARGO_PKG_VERSION`.

### Added
- `predict --energy [--num-gpus N]` — energy (J/kWh) and CO2 estimates,
  wiring the existing `EnergyModel` into the CLI.
- `predict --quantum <hardware> [--precision P]` — quantum fidelity, shot
  count, and execution-time prediction (`superconducting`, `trapped_ion`,
  `neutral_atom`), wiring the existing `predict_quantum` into the CLI.
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, and an issue-template chooser
  (`.github/ISSUE_TEMPLATE/config.yml`). Private vulnerability reporting is
  now enabled on the repository so `SECURITY.md`'s instructions work.

### Changed
- Removed 25 declared-but-unused dependencies across the workspace (found
  with `cargo-machete`, each verified by hand before removal).
- Eliminated needless `Vec` collects and redundant clones on `lift-opt`'s
  hot paths (flash-attention, quantisation-pass, real-routing).
- `lift-test/` (root) moved to `crates/lift-demo/` — it was the only
  workspace member outside `crates/`, and its name was one character from
  the unrelated `crates/lift-tests` integration-test crate.
- Consolidated secondary docs (`CAPABILITIES.md`, `DIALECTS.md`,
  `LIFT_design.md`, `LIFT_Guide.md`, `LIFT_Manual.md`, `PUBLISHING.md`,
  `STRATEGY.md`) into `docs/`; `README.md`, `LICENSE`, `CHANGELOG.md`, and
  `CONTRIBUTING.md` stay at the root.
- Translated `docs/CAPABILITIES.md` from French to English (it was the last
  fully-French document in the project) and corrected several claims that had
  gone stale since it was written, including two caught by this release's own
  fixes (QASM qubit indexing, gate-decomposition).

## [0.4.4] — 2026-08-05

### Changed
- **Automated releases via crates.io Trusted Publishing (OIDC)** — no API
  token needed. All 13 crates configured with `rustnew/Lift` workflow
  `publish.yml`; pushing a `v*` tag publishes every crate in dependency order
  from CI (`.github/workflows/publish.yml`).
- Version bump 0.4.3 → 0.4.4 across workspace and docs.

## [0.4.3] — 2026-08-05

### Changed
- Optimised crate descriptions for discoverability: every description now
  leads with "LIFT compiler", so the crates surface in crates.io searches for
  "compiler", "compiler framework", "quantum compiler", and "AI compiler".
- All 13 crates republished to crates.io at v0.4.3.

## [0.4.2] — 2026-08-05

### Fixed
- **LICENSE now ships in every published crate package** (was missing from
  crates.io tarballs because Cargo only auto-includes LICENSE files located in
  each package directory, not the workspace root).
- **Repository field corrected to `rustnew/Lift`** in all published manifests
  (the GitHub rename from `Litf-IR` had not been propagated to crates.io).
- Docs version references bumped to 0.4.2.

### Changed
- All 13 crates republished to crates.io at v0.4.2.

## [0.4.1] — 2026-08-05

### Fixed
- Corrected op/gate counts in docs (110 tensor ops, 48 quantum gates, 21 hybrid ops).
- README examples now compile against the real API (`GateDecomposition::new(Provider::IbmKyoto)`, `DataType` re-export from `model_builder`).
- Repository references updated to `rustnew/Lift` (renamed from `Litf-IR`).

### Changed
- Architecture diagrams moved to Mermaid (pipeline, dependency layers, roadmap).
- All 13 crates republished to crates.io at v0.4.1.

## [0.4.0] — 2026-08-05

### Added
- **Optimisation levels `O0`–`O3`** with explicit-pass override and per-pass enable/disable.
- **Semantic verification** (op arity vs dialect signatures).
- **13 optimisation passes** including generic tensor fusion, hardware-native gate
  decomposition, real qubit routing (SWAP + BFS), non-adjacent gate cancellation &
  rotation merging.
- All 13 crates published to crates.io (first full workspace release).

## [0.3.0] — 2026-04-30

### Added
- Tensor / quantum / hybrid dialects.
- Cost modelling (FLOPs, memory, energy/carbon).
- Performance prediction (roofline analysis).
- Export backends (LLVM IR, ONNX, OpenQASM 3.0).

## [0.2.1] — 2026-04-30

### Changed
- Stability tuning.

## [0.2.0] — 2026-03-31

### Added
- Initial public release of the LIFT compiler framework.
- SSA-based intermediate representation.
- Tensor, quantum, and hybrid dialects.
- Core compiler infrastructure (types, values, operations, blocks, regions, verifier).