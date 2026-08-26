<div align="center">

<img width="1262" height="602" alt="LIFT Framework" src="https://github.com/user-attachments/assets/3880ecec-ff3f-4b44-b256-c3a9f07ee813" />

# LIFT

**Language for Intelligent Frameworks and Technologies**

The first Intermediate Representation built natively for both AI and Quantum Computing.

*Simulate before you run. Compile once. Optimise everywhere.*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](../LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://rustlang.org)
[![Tests](https://img.shields.io/badge/Tests-541%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/Version-0.4.7-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Alpha-gold.svg)]()

</div>

---

## Overview

LIFT is a **unified compiler infrastructure** that treats AI computation (tensors, gradients, attention) and quantum computation (qubits, gates, noise models) as first-class citizens in the same SSA-based intermediate representation. One `.lif` source file, one `.lith` config, one target pipeline: **simulate, predict, optimise, compile**.

That target is a work in progress, not today's state — see
[docs/CAPABILITIES.md](CAPABILITIES.md) for an honest, source-verified
breakdown of what's real versus planned for each of the four stages.
**Predict** and **Optimise** are solid; **Simulate** is static analysis only
(no real execution yet); **Compile** produces text output, not executable
code, for any target.

```
 .lif source ──► LIFT-CORE (SSA IR) ──► SIMULATE ──► PREDICT ──► OPTIMISE ──► COMPILE
                      │                    (static)                              │
          ┌───────────┼───────────┐                                 ┌────────────┼────────────┐
     LIFT-TENSOR  LIFT-QUANTUM  LIFT-HYBRID                   OpenQASM 3   LLVM IR text   ONNX
     110 tensor   48 gates     21 hybrid                     (48/48 gates)  (skeleton)  (opset 21)
     operations   Kraus/QEC     VQC/VQE ops                                              CUDA PTX (planned)
```

---

## Why LIFT?

No existing IR handles both AI and quantum in a single representation.

| Capability | MLIR | ONNX | OpenQASM | Qiskit | **LIFT** |
|---|:---:|:---:|:---:|:---:|:---:|
| AI tensor operations | Y | Y | - | - | **Y** |
| Quantum gate operations | - | - | Y | Y | **Y** |
| Unified AI + Quantum IR | - | - | - | ~ | **Y** |
| Noise as type-level attribute | - | - | - | - | **Y** |
| Linear qubit types (no-cloning) | - | - | - | - | **Y** |
| Budget enforcement before compile | - | - | - | - | **Y** |
| Single config for entire pipeline | - | - | - | - | **Y** |
| Performance prediction engine | - | - | - | - | **Y** |

**Key:** Y = implemented, ~ = partial, - = not supported

### What makes LIFT unique

1. **One IR for AI + Quantum** -- Both are equal citizens in the same SSA graph. Joint optimisation across classical and quantum operations.
2. **Noise in the type system** -- Every quantum gate carries T1/T2, fidelity, crosstalk metadata. The compiler reasons about noise at every stage.
3. **Linear qubit types** -- The no-cloning theorem enforced at compile time. Double-use of a qubit is a type error, not a runtime crash.
4. **Simulation-first compilation** -- FLOP count, peak memory, circuit depth, expected fidelity, energy cost -- all computed *before* hardware runs. Budget violations halt compilation with actionable suggestions.
5. **One config language** -- The `.lith` file replaces 6-8 separate configuration files across frameworks.

---

## Architecture

```
  USER        .lif source  |  .lith config  |  lift(1) CLI
  FRONTEND    Lexer > Parser > AST > SSA Builder  |  Importers: ONNX, PyTorch FX, OpenQASM 3
  DIALECTS    LIFT-CORE  |  LIFT-TENSOR  |  LIFT-QUANTUM  |  LIFT-HYBRID
  ANALYSIS    Shape inference  |  FLOP count  |  Noise sim  |  Energy model  |  Roofline
  PASSES      TensorFusion  FlashAttention  GateCancellation  RotationMerge  LayoutMapping  CSE ...
  BACKENDS    OpenQASM 3 (48/48 gates)  |  LLVM IR (skeleton)  |  ONNX (opset 21)  |  CUDA PTX, XLA (planned)
  HARDWARE    H100 / A100 / MI300  |  IBM Kyoto / Rigetti / IonQ  |  TPU
```

### Crate Map

| Crate | Purpose | Key contents |
|-------|---------|-------------|
| `lift-core` | SSA IR foundation | Types, values, operations, blocks, regions, verifier, printer, pass manager |
| `lift-ast` | Frontend | Lexer, parser, AST, IR builder for `.lif` files |
| `lift-tensor` | AI dialect | 110 ops (attention, conv, pooling, MoE, quantisation, GNN, fused), shape inference |
| `lift-quantum` | Quantum dialect | 48 gates (IBM/Rigetti/IonQ native), noise models, Kraus channels, QEC, topology |
| `lift-hybrid` | Fusion dialect | 21 ops (VQC, VQE, QAOA), gradient methods, encoding strategies, GPU-QPU transfer |
| `lift-sim` | Analysis engine | Cost models (A100/H100), quantum cost (superconducting/trapped-ion/neutral-atom), energy, carbon |
| `lift-predict` | Prediction | Roofline model, budget enforcement |
| `lift-opt` | Optimisation | 13 passes: DCE, constant fold, tensor fusion, flash attention, gate cancel, rotation merge, CSE, quantisation, noise-aware schedule, layout mapping, canonicalise, gate decomposition, real routing |
| `lift-import` | Importers | ONNX, PyTorch FX, OpenQASM 3 |
| `lift-export` | Backends | LLVM IR, ONNX (opset 21), OpenQASM 3 |
| `lift-config` | Configuration | `.lith` parser and validator |
| `lift-cli` | CLI | `lift verify`, `lift analyse`, `lift print`, `lift optimise`, `lift predict`, `lift export` |
| `lift-codegen` | Codegen | Programmatic model generation, multi-format export |

---

## Quick Start

```bash
# Install Rust 1.80+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/rustnew/Lift.git
cd Lift
cargo build --release

# Run tests (541 tests)
cargo test --workspace

# Install the `lift` CLI on your PATH (or use `cargo run -p lift-cli --` instead)
cargo install lift-cli
```

### Example: Tensor Program

```bash
cat > hello.lif << 'EOF'
#dialect tensor
module @test {
    func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {
        %out = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %out
    }
}
EOF

lift verify  hello.lif    # Check IR well-formedness
lift analyse hello.lif    # FLOPs, shapes, memory estimate
lift print   hello.lif    # Pretty-print the IR
```

### Example: Quantum Circuit

```lif
#dialect quantum
module @bell {
    func @bell_state(%q0: qubit, %q1: qubit) -> (bit, bit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        %b0 = "quantum.measure"(%q3) : (qubit) -> bit
        %b1 = "quantum.measure"(%q4) : (qubit) -> bit
        return %b0, %b1
    }
}
```

(Every `%name` is assigned exactly once — SSA requires this. Reusing `%q0` as
both the block argument and a gate result, as an earlier version of this
example did, fails `lift verify` with `MultipleDefinition`.)

---

## The `.lith` Configuration

One file controls the entire compilation pipeline:

```lith
[target]
backend = "llvm"
device = "h100"
precision = "fp16"

[quantum]
provider = "ibm_kyoto"
topology = "heavy_hex"
num_qubits = 27

[optimisation]
level = O3
passes = canonicalize, tensor-fusion, gate-cancellation, gate-decomposition, real-routing

[budget]
max_memory_bytes = 80000000000
max_time_ms = 200.0
min_fidelity = 0.92
```

---

## Optimisation Passes

All 13 passes are reachable from the CLI and from `.lith`'s `[optimisation] passes = ...`.

| Pass | Domain | Description |
|------|--------|-------------|
| Canonicalise | All | Normalise IR to canonical form |
| Constant Folding | All | Evaluate compile-time constants |
| Dead Code Elimination | All | Remove unused operations |
| Common Subexpression Elimination | All | Deduplicate identical computations |
| Tensor Fusion | AI | Fuse MatMul+Bias+ReLU, Linear+GELU/SiLU, Conv+BN+ReLU chains |
| Flash Attention | AI | Replace standard attention with FlashAttention above a sequence-length threshold |
| Quantisation | AI | Annotate compute-heavy ops for INT8/INT4/FP8 quantisation |
| Gate Cancellation | Quantum | Cancel H·H=I, X·X=I, S·Sdg=I, T·Tdg=I, including non-consecutive pairs |
| Rotation Merge | Quantum | Merge Rz(a)·Rz(b) → Rz(a+b), including non-consecutive pairs |
| Noise-Aware Schedule | Quantum | Reorder gates to minimise decoherence |
| Layout Mapping | Quantum | Legacy pass: annotates non-adjacent 2-qubit gates with `needs_swap = true` — does not insert SWAPs itself |
| Gate Decomposition | Quantum | Replace H/T/Tdg/S/Sdg/Y/RX with the target provider's native gate set |
| Real Routing | Quantum | Insert real `quantum.swap` ops (BFS shortest path) so 2-qubit gates land on connected physical qubits |

---

## Current Status

| Component | Status | Coverage |
|-----------|--------|----------|
| `lift-core` | Stable | SSA IR, types, verifier, printer, pass manager |
| `lift-ast` | Stable | Full lexer, parser, AST, IR builder |
| `lift-tensor` | Stable | 110 operations, shape inference, FLOP counting |
| `lift-quantum` | Stable | 48 gates, noise models, Kraus channels, QEC codes, topology |
| `lift-hybrid` | Stable | 21 operations, gradient methods, encoding strategies |
| `lift-sim` | Stable | Cost models, energy model, quantum simulation, budget tracking |
| `lift-predict` | Stable | Roofline model, budget enforcement |
| `lift-opt` | Stable | 13 optimisation passes |
| `lift-import` | Skeleton | ONNX/PyTorch FX/OpenQASM 3 importers parse the source format but don't yet convert nodes into LIFT ops |
| `lift-export` | Active | ONNX (opset 21) is operational; OpenQASM covers all 48 gates (46 as real instructions, 2 as comments); LLVM IR is a text skeleton (ops as comments) |
| `lift-config` | Stable | `.lith` parser and types |
| `lift-cli` | Stable | verify, analyse, print, optimise, predict, export |
| `lift-codegen` | Stable | programmatic model generation, multi-format export |

**Test suite:** 541 tests, 100% pass rate across 14 crates.

---

## Roadmap

| Phase | Target | Milestone |
|-------|--------|-----------|
| Core IR + Dialects | Done | SSA IR, tensor/quantum/hybrid dialects complete |
| Optimisation Passes | Done | 13 passes implemented and tested |
| Analysis Engine | Done | Cost models, energy, noise simulation |
| Functional Import/Export | Planned (v0.5) | Real ONNX/PyTorch FX/OpenQASM import; full 50+-gate OpenQASM export |
| Hardware Backends | Planned | CUDA PTX, native OpenQASM execution |
| Python Bindings | Planned | PyO3-based Python API |
| v1.0 Release | Q4 2026 | Full pipeline, benchmarks, arXiv paper |

---

## Contributing

| Area | Difficulty | Description |
|------|-----------|-------------|
| CUDA PTX backend | Hard | GPU code generation for tensor ops |
| State vector simulator | Medium | Quantum circuit simulator (CPU + GPU) |
| Qiskit importer | Medium | Import Qiskit circuits into LIFT IR |
| API documentation | Easy | Rustdoc for all public items |
| Tutorials | Easy | Getting started guides and examples |

See [CONTRIBUTING.md](../CONTRIBUTING.md) for code style and PR process.

---

## Citation

```bibtex
@software{lift2025,
  title  = {LIFT: Language for Intelligent Frameworks and Technologies},
  author = {LIFT Framework Contributors},
  year   = {2025},
  url    = {https://github.com/rustnew/Lift},
  note   = {Unified IR for AI and Quantum Computing}
}
```

## License

MIT -- see [LICENSE](../LICENSE).

---

<div align="center">

*LIFT -- Because the future of computation is both intelligent and quantum, and it deserves a unified foundation.*

</div>