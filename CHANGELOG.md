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