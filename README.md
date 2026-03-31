<div align="center">

<img width="1262" height="602" alt="LIFT Framework" src="https://github.com/user-attachments/assets/3880ecec-ff3f-4b44-b256-c3a9f07ee813" />

# LIFT

**Language for Intelligent Frameworks and Technologies**

The first Intermediate Representation built natively for both AI and Quantum Computing.

*Simulate before you run. Compile once. Optimise everywhere.*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80+-orange.svg)](https://rustlang.org)
[![Tests](https://img.shields.io/badge/Tests-505%20passed-brightgreen.svg)]()
[![Version](https://img.shields.io/badge/Version-0.2.0-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Alpha-gold.svg)]()

</div>

---

## Overview

LIFT is a **unified compiler infrastructure** that treats AI computation (tensors, gradients, attention) and quantum computation (qubits, gates, noise models) as first-class citizens in the same SSA-based intermediate representation. One `.lif` source file, one `.lith` config, one pipeline: **simulate, predict, optimise, compile**.

```
 .lif source ──► LIFT-CORE (SSA IR) ──► SIMULATE ──► PREDICT ──► OPTIMISE ──► COMPILE
                      │                                                          │
          ┌───────────┼───────────┐                                 ┌────────────┼────────────┐
     LIFT-TENSOR  LIFT-QUANTUM  LIFT-HYBRID                    CUDA (GPU)   OpenQASM 3   LLVM (CPU)
     90+ tensor   50+ gates     21 hybrid                      H100/A100    IBM/Rigetti   AVX-512
     operations   Kraus/QEC     VQC/VQE ops                    MI300        IonQ          OpenMP
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
  BACKENDS    CUDA (PTX)  |  OpenQASM 3  |  LLVM IR  |  XLA (planned)
  HARDWARE    H100 / A100 / MI300  |  IBM Kyoto / Rigetti / IonQ  |  TPU
```

### Crate Map

| Crate | Purpose | Key contents |
|-------|---------|-------------|
| `lift-core` | SSA IR foundation | Types, values, operations, blocks, regions, verifier, printer, pass manager |
| `lift-ast` | Frontend | Lexer, parser, AST, IR builder for `.lif` files |
| `lift-tensor` | AI dialect | 90+ ops (attention, conv, pooling, MoE, quantisation, GNN, fused), shape inference |
| `lift-quantum` | Quantum dialect | 50+ gates (IBM/Rigetti/IonQ native), noise models, Kraus channels, QEC, topology |
| `lift-hybrid` | Fusion dialect | 21 ops (VQC, VQE, QAOA), gradient methods, encoding strategies, GPU-QPU transfer |
| `lift-sim` | Analysis engine | Cost models (A100/H100), quantum cost (superconducting/trapped-ion/neutral-atom), energy, carbon |
| `lift-predict` | Prediction | Roofline model, budget enforcement |
| `lift-opt` | Optimisation | 11 passes: DCE, constant fold, tensor fusion, flash attention, gate cancel, rotation merge, CSE, quantisation, noise-aware schedule, layout mapping, canonicalise |
| `lift-import` | Importers | ONNX, PyTorch FX, OpenQASM 3 |
| `lift-export` | Backends | LLVM IR, OpenQASM 3 |
| `lift-config` | Configuration | `.lith` parser and validator |
| `lift-cli` | CLI | `lift verify`, `lift analyse`, `lift print`, `lift optimise`, `lift export` |

---

## Quick Start

```bash
# Install Rust 1.80+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/lift-framework/lift
cd lift
cargo build --release

# Run tests (505 tests)
cargo test --workspace
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
    func @bell_state() -> (bit, bit) {
        %q0 = "quantum.init"() : () -> qubit
        %q1 = "quantum.init"() : () -> qubit
        %q0 = "quantum.h"(%q0)         : (qubit) -> qubit
        %q0, %q1 = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)
        %b0 = "quantum.measure"(%q0) : (qubit) -> bit
        %b1 = "quantum.measure"(%q1) : (qubit) -> bit
        return %b0, %b1
    }
}
```

---

## The `.lith` Configuration

One file controls the entire compilation pipeline:

```lith
compilation {
    target {
        gpu  { backend = "cuda"  arch = "sm_90"  memory_limit_gb = 80 }
        qpu  { provider = "ibm"  backend_name = "ibm_kyoto"  shots = 4096 }
    }
}
optimization {
    pipeline = ["canonicalize", "tensor-fusion", "gate-cancellation", "layout-mapping"]
}
prediction {
    budget { max_latency_ms = 200  min_fidelity = 0.92  max_memory_gb = 40 }
}
```

---

## Optimisation Passes

| Pass | Domain | Description |
|------|--------|-------------|
| Canonicalise | All | Normalise IR to canonical form |
| Constant Folding | All | Evaluate compile-time constants |
| Dead Code Elimination | All | Remove unused operations |
| Tensor Fusion | AI | Fuse MatMul+Bias+ReLU chains (30-50% bandwidth reduction) |
| Flash Attention | AI | Replace O(n^2) attention with tiled O(n) (10-20x speedup) |
| Quantisation | AI | INT8/FP8 annotation (4x model size reduction) |
| Common Subexpression Elimination | All | Deduplicate identical computations |
| Gate Cancellation | Quantum | H*H=I, Rz(a)*Rz(b)=Rz(a+b) (15-40% depth reduction) |
| Rotation Merge | Quantum | Merge consecutive rotation gates |
| Noise-Aware Schedule | Quantum | Reorder gates for maximum fidelity |
| Layout Mapping | Quantum | SABRE routing to physical qubit topology |

---

## Current Status

| Component | Status | Coverage |
|-----------|--------|----------|
| `lift-core` | Stable | SSA IR, types, verifier, printer, pass manager |
| `lift-ast` | Stable | Full lexer, parser, AST, IR builder |
| `lift-tensor` | Stable | 90+ operations, shape inference, FLOP counting |
| `lift-quantum` | Stable | 50+ gates, noise models, Kraus channels, QEC codes, topology |
| `lift-hybrid` | Stable | 21 operations, gradient methods, encoding strategies |
| `lift-sim` | Stable | Cost models, energy model, quantum simulation, budget tracking |
| `lift-predict` | Stable | Roofline model, budget enforcement |
| `lift-opt` | Stable | 11 optimisation passes |
| `lift-import` | Active | ONNX, PyTorch FX, OpenQASM 3 importers |
| `lift-export` | Active | LLVM IR, OpenQASM 3 exporters |
| `lift-config` | Stable | `.lith` parser and types |
| `lift-cli` | Stable | verify, analyse, print, optimise, export |

**Test suite:** 505 tests, 100% pass rate across 12 crates.

---

## Roadmap

| Phase | Target | Milestone |
|-------|--------|-----------|
| Core IR + Dialects | Done | SSA IR, tensor/quantum/hybrid dialects complete |
| Optimisation Passes | Done | 11 passes implemented and tested |
| Analysis Engine | Done | Cost models, energy, noise simulation |
| Import/Export | Active | ONNX, PyTorch FX, LLVM, OpenQASM |
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and PR process.

---

## Citation

```bibtex
@software{lift2025,
  title  = {LIFT: Language for Intelligent Frameworks and Technologies},
  author = {Martial-Christian and Contributors},
  year   = {2025},
  url    = {https://github.com/lift-framework/lift},
  note   = {Unified IR for AI and Quantum Computing}
}
```

## License

MIT -- see [LICENSE](LICENSE).

---

<div align="center">

*LIFT -- Because the future of computation is both intelligent and quantum, and it deserves a unified foundation.*

</div>