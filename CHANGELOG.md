# Changelog

All notable changes to LIFT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (LIFT v5 roadmap)
- Functional ONNX / PyTorch FX / OpenQASM importers (currently stubs)
- Real LLVM IR lowering with cuBLAS/cuDNN runtime calls
- State-vector quantum simulator (CPU, up to ~25 qubits)
- Tensor interpreter for simulation-first execution
- SABRE qubit routing with dynamic re-placement
- Multi-file support (`include` / linking)
- True automatic differentiation (backward graph construction)
- PyO3 Python bindings

## [0.4.0] - 2026

### Added
- **Optimisation pipeline by level** (`lift-config`): `O0`–`O3` presets,
  `OptimisationConfig::passes_for_level()`, `effective_passes()` with
  explicit-passes override and `disabled_passes`, unknown-pass detection
  (`OptimisationConfig::validate()`). 13 passes registered in `ALL_PASSES`.
- **Semantic verification** (`lift-core`): `verify_semantics()` checks operation
  input arity against dialect signatures; public `verify_with_dialects()` lets
  callers supply a full dialect registry; `cmd verify` now builds the complete
  registry (core + tensor + quantum + hybrid).
- **Generic fusion pass** (`lift-opt`): 5 patterns — `matmul+bias+relu`,
  `matmul+bias`, `linear+gelu`, `linear+silu`, `conv2d+bn+relu` — in a two-phase
  runner (ternary patterns before binary ones).
- **Gate decomposition** (`lift-opt`): `gate-decomposition` pass lowering H, T,
  Tdg, S, Sdg, Y, RX to native gate sets (IBM/Rigetti/IonQ/Quantinuum) driven by
  `QuantumConfig.provider`. New `QuantumProvider` enum and `provider` key in
  `.lith` files.
- **Non-adjacent gate cancellation & rotation merging** (`lift-opt`): passes now
  cancel/merge gate pairs that are separated by commuting gates on other qubits
  (SSA-chain check), not just consecutive ones.
- **Real qubit routing** (`lift-opt`): `real-routing` pass inserts actual
  `quantum.swap` operations (BFS shortest path) to satisfy device connectivity,
  with logical↔physical placement tracking. Topology from `QuantumConfig`
  (linear/grid/heavy-hex/all-to-all/tree).
- `Context::insert_op_before()` preserves SSA dominance when passes insert ops.
- Test count grows to **535** (was 515); `examples/validate_all.sh` now covers
  all 13 passes and reports **105/105**.

### Fixed
- `resnet_generated.lif` called `tensor.batchnorm` with 2 inputs (minimum 3);
  `lift-codegen` now emits the scale/bias parameters (`bn1_b`, `bn2_b`).
- `examples/validate_all.sh` referenced removed `deepseek_v2_lite.lif` and
  `attention.lif` models; script updated to current model set.
- Remaining dead-code warning in `lift-export` LLVM backend resolved
  (`llvm_type_for_value` now annotates emitted IR).
- Workspace, all crates, and CLI aligned on version **0.4.0**.

## [0.3.0] - 2026

### Added
- `lift-tests` integration crate (515 tests, 0 failures)
- `lift-test` hybrid AI+Quantum integration test (CNN + VQC medical imaging,
  17 pipeline steps)
- ONNX export backend (opset 21, protobuf text, standard + com.microsoft ops)
- `lift-codegen` binary: programmatic model generation (Phi-3, MLP, ResNet, VQE)
  with multi-format export (`.lif`, `.ll`, `.onnx`, `.qasm`, `.lith`)
- `ModelBuilder` fluent API in `lift-core`
- 11 optimisation passes (canonicalize, constant folding, DCE, CSE, tensor
  fusion, flash attention, quantisation, gate cancellation, rotation merge,
  noise-aware schedule, layout mapping)
- Energy/carbon estimation models (`EnergyModel`)
- Reactive budget tracking (`ReactiveBudget`)
- Device topologies (grid, heavy-hex, all-to-all, linear, tree) with BFS
- QEC codes (Surface, Steane, Shor, Repetition, LDPC)
- Kraus channel noise models (6 channels)
- Complete documentation: `README.md`, `DIALECTS.md`, `CAPABILITIES.md`,
  `LIFT_Guide.md`, `LIFT_Manual.md`, `LIFT_design.md`, `STRATEGY.md`,
  `MANUAL.md`, `CONTRIBUTING.md`, `CHANGELOG.md`

### Fixed
- CLI version aligned with workspace version
- Clippy lint errors (approximate constants, formatting)
- Codebase fully `rustfmt`-clean

## [0.2.0] - 2026

### Added
- Core SSA IR (`lift-core`): `Context`, type system, verifier, printer,
  pass manager, dialect registry
- `.lif` frontend (`lift-ast`): lexer, parser, IR builder
- Tensor dialect (`lift-tensor`) with shape inference and FLOP counting
- Quantum dialect (`lift-quantum`): gates, providers, noise, topology
- Hybrid dialect (`lift-hybrid`): encoding, gradients, variational algorithms
- Static analysis (`lift-sim`) and roofline prediction (`lift-predict`)
- 6 optimisation passes
- LLVM IR and OpenQASM 3.0 export backends
- `.lith` configuration parser (`lift-config`)
- `lift` CLI: verify, analyse, print, optimise, predict, export

## [0.1.0] - 2025

### Added
- Initial project scaffolding and design documentation
- Prototype IR types and verifier

[Unreleased]: https://github.com/lift-lang/lift
[0.3.0]: https://github.com/lift-lang/lift/releases/tag/v0.3.0
[0.2.0]: https://github.com/lift-lang/lift/releases/tag/v0.2.0
[0.1.0]: https://github.com/lift-lang/lift/releases/tag/v0.1.0
