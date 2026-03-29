<img width="1262" height="602" alt="image" src="https://github.com/user-attachments/assets/3880ecec-ff3f-4b44-b256-c3a9f07ee813" />

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ██╗     ██╗███████╗████████╗                                      ║
║    ██║     ██║██╔════╝╚══██╔══╝                                      ║
║    ██║     ██║█████╗     ██║                                         ║
║    ██║     ██║██╔══╝     ██║                                         ║
║    ███████╗██║██║        ██║                                         ║
║    ╚══════╝╚═╝╚═╝        ╚═╝                                         ║
║                                                                      ║
║    Language for Intelligent Frameworks and Technologies              ║
║    ─────────────────────────────────────────────────────             ║
║    AI  ·  Quantum  ·  Hybrid  ·  Unified IR                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

**The first Intermediate Representation built natively for both AI and Quantum Computing.**

*Simulate before you run. Compile once. Optimise everywhere.*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)
[![Rust 1.78+](https://img.shields.io/badge/Language-Rust%201.78+-orange.svg)](https://rustlang.org)
[![Phase 0–1 Active](https://img.shields.io/badge/Phase-0--1%20Active-red.svg)]()
[![Research Alpha](https://img.shields.io/badge/Status-Research%20Alpha-gold.svg)]()
[![arXiv](https://img.shields.io/badge/Paper-arXiv%20draft-orange.svg)]()

</div>

---

> **⚠ HONEST STATUS — read this first**
>
> This document presents the complete vision and architecture for LIFT.
> **Phase 0 (LIFT-CORE) is complete. Phase 1 (LIFT-TENSOR) is in active development.**
> Quantum and hybrid support are designed but not yet implemented.
> Python bindings exist as a scaffold. Do not use in production.
>
> We publish the full vision now because architectural decisions
> must be made correctly from day one. See [Section 9](#9-current-status) for what works today.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Vision](#2-the-vision)
3. [Honest Comparison with Existing IRs](#3-honest-comparison)
4. [Core Concept — Twin Dialects](#4-twin-dialects)
5. [Full Architecture](#5-architecture)
6. [The Four Pillars](#6-the-four-pillars)
7. [The .lif Source Language](#7-the-lif-language)
8. [The .lith Configuration](#8-the-lith-configuration)
9. [Current Status](#9-current-status)
10. [Getting Started](#10-getting-started)
11. [Roadmap — 24 Months](#11-roadmap)
12. [Contributing](#12-contributing)
13. [Known Limitations](#13-known-limitations)
14. [Why This Matters](#14-why-this-matters)

---

## 1. The Problem

### Fragmentation of the Toolchain

A researcher working on hybrid AI+Quantum computing today must master and coordinate eight incompatible tools:

```
  AI WORLD                                QUANTUM WORLD
  ──────────────────────────              ──────────────────────────
  PyTorch  ──┐                            Qiskit    ──┐
  JAX      ──┼──► MLIR / ONNX / XLA       Cirq      ──┼──► OpenQASM 3
  TF       ──┘         │                  PennyLane ──┘       │
                       ▼                                      ▼
              GPU / TPU / CPU                     IBM Q / Rigetti / IonQ

  ✗  No shared representation             ✗  No shared representation
  ✗  No joint optimisation                ✗  Cannot compose with AI
  ✗  Energy cost invisible                ✗  Noise is an afterthought
  ✗  8+ config files per project          ✗  No simulation-first workflow
  ✗  Performance surprises at runtime     ✗  No budget enforcement
```

### The Scale Problem

```
  AI MODEL SIZE (parameters)
  ──────────────────────────────────────────────────────────────────
  2018  BERT-Large    ██  340 M
  2020  GPT-3         ████████████████  175 B
  2023  GPT-4 (est.)  ████████████████████████  ~1.7 T
  2025  Next wave     ████████████████████████████████  10 T+

  10 T parameters in FP16 = 20 TB.  No single IR handles this today.

  QUANTUM HARDWARE (physical qubits)
  ──────────────────────────────────────────────────────────────────
  2019  Google Sycamore  ██  53
  2023  IBM Eagle         ████████████  433
  2026  Target            ████████████████████  1 000+

  Both worlds are scaling fast. The unified toolchain does not exist.
```

---

## 2. The Vision

LIFT is a **unified semantic Intermediate Representation** that understands both AI computation (tensors, gradients, attention mechanisms) and quantum computation (qubits, gates, decoherence, noise models) in the same programme, compiled by the same pipeline, governed by the same configuration file.

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Your programme   (one .lif file)                          ║
  ║                                                              ║
  ╠═══════════════╦═════════════════════════════════════════════╣
  ║  LIFT-TENSOR  ║  LIFT-QUANTUM           LIFT-HYBRID         ║
  ║  AI dialect   ║  Quantum dialect        Fusion dialect      ║
  ╠═══════════════╩═════════════════════════════════════════════╣
  ║                                                              ║
  ║    SIMULATE  →  PREDICT  →  OPTIMISE  →  COMPILE            ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  CUDA (GPU)  OpenQASM 3 (QPU)  LLVM (CPU)  XLA (TPU)       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  H100 · A100 · MI300  │  IBM Kyoto · Rigetti · IonQ         ║
  ╚══════════════════════════════════════════════════════════════╝
```

**The north-star metric:** A researcher goes from idea to optimised hybrid execution on real hardware in under one hour — one `.lif` source file, one `.lith` config. Today that takes weeks.

---

## 3. Honest Comparison

`~✓` means planned and designed, not yet implemented. We do not overstate.

```
  ┌────────────────────────┬──────┬──────┬──────────┬────────┬─────────────┐
  │ Capability             │ MLIR │ ONNX │ OpenQASM │ Qiskit │    LIFT     │
  ├────────────────────────┼──────┼──────┼──────────┼────────┼─────────────┤
  │ AI tensor operations   │  ✓   │  ✓   │    ✗     │   ✗    │   ✓ stable  │
  │ Quantum gate ops       │  ✗   │  ✗   │    ✓     │   ✓    │  ~✓ dev     │
  │ Hybrid AI+QC in one IR │  ✗   │  ✗   │    ✗     │  ~✓    │  ~✓ planned │
  │ Noise in type system   │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ planned │
  │ Linear qubit types     │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ dev     │
  │ Perf. prediction (GNN) │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ dev     │
  │ Energy budgeting       │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ planned │
  │ Single config file     │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ dev     │
  │ Python bindings        │  ✓   │  ✓   │   ~✓     │   ✓    │  ~✓ dev     │
  ├────────────────────────┼──────┼──────┼──────────┼────────┼─────────────┤
  │ Score today            │ 3/9  │ 2/9  │  3/9     │ 3/9    │  2/9 ✓      │
  │ Score at v1.0          │      │      │          │        │  8/9 ~✓     │
  └────────────────────────┴──────┴──────┴──────────┴────────┴─────────────┘

  ✓   = implemented, stable today
  ~✓  = in design or active development, not yet stable
  ✗   = not supported
```

### What LIFT Adds That Does Not Exist Anywhere Today

**1. One IR for AI and quantum in the same programme.**
MLIR has experimental quantum dialect work (QSSA, Catalyst). None treat noise as a first-class type attribute. None provide joint optimisation between tensor and quantum operations. LIFT is the first design where both are equal citizens in the same SSA IR.

**2. Noise as a type-level attribute.**
Every quantum gate carries optional noise metadata: T1, T2, gate fidelity, crosstalk coefficients. The type checker, optimiser, and predictor all reason over this noise. When two noisy gates are fused, the composite noise is derived (depolarising approximation in v1.0, full Kraus operators in v1.1).

**3. Linear qubit types — no-cloning enforced at compile time.**
The quantum no-cloning theorem is a physical law: quantum information cannot be copied. LIFT enforces this as a compile-time type error. A qubit used twice is caught before hardware execution, not after. Every branch arm must consume the same qubit set.

**4. Simulation-driven compilation with budget enforcement.**
Before any hardware executes: FLOP count, peak memory, circuit depth, expected fidelity, estimated latency, energy cost. If any budget constraint is violated, compilation fails with an actionable error and suggestions. This is architecturally different from post-hoc profiling.

**5. One configuration language for the entire pipeline.**
The `.lith` file controls compilation target, optimisation passes, budget constraints, simulation parameters, deployment, and monitoring — replacing 6–8 separate configs.

---

## 4. Twin Dialects

### The Structural Isomorphism

The deepest insight in LIFT: AI and quantum compilation face the same class of problems, with different vocabulary. LIFT exploits this with twin dialects built on a shared SSA foundation.

```
  AI DOMAIN                              QUANTUM DOMAIN
  ════════════════════════════════════════════════════════════
  Tensor (float vector)           ↔   Quantum state (amplitude vector)
  Linear layer (matrix multiply)  ↔   Unitary gate (unitary multiply)
  Non-linearity (ReLU)            ↔   Measurement (projection, collapse)
  Backpropagation (reverse AD)    ↔   Parameter shift rule (adjoint diff)
  Batch dimension                 ↔   Shot parallelism
  INT8 quantisation               ↔   Gate decomposition to native basis
  Layer fusion (MatMul+ReLU)      ↔   Gate cancellation (H·H = I)
  Memory layout (NCHW vs NHWC)    ↔   Qubit mapping (logical → physical)
  Multi-GPU data parallelism      ↔   Multi-QPU shot parallelism
  Gradient checkpoint             ↔   Mid-circuit reset and reuse
  ────────────────────────────────────────────────────────────
  Same class of problems. Different vocabulary.
  → LIFT exploits this isomorphism for joint optimisation.
```

### SSA Form — The Shared Foundation

Every value in LIFT is defined exactly once (Static Single Assignment). This makes analysis and optimisation provably correct.

```
  TRADITIONAL              LIFT SSA FORM
  ─────────────────        ──────────────────────────────────────────
  x = matmul(A, B)         %v0 = tensor.matmul(%A, %B)
  x = relu(x)       →      %v1 = tensor.relu(%v0)
  x = layernorm(x)         %v2 = tensor.layernorm(%v1, %w, %b)

                            Each %vi defined ONCE
                            → safe to fuse, reorder, parallelise
```

### Linear Types — No-Cloning Enforced

```
  ✗ FORBIDDEN                          ✓ CORRECT
  ─────────────────────────────        ────────────────────────────────
  %q0 = quantum.init() : qubit         %q0 = quantum.init() : qubit
  %q1 = quantum.x(%q0)  : qubit        %q1 = quantum.x(%q0)  : qubit
  %q2 = quantum.h(%q0)  ← ERROR        %q2 = quantum.h(%q1)  : qubit
                                        %b0 = quantum.meas(%q2) : bit
  TYPE ERROR: %q0 already consumed
  by quantum.x. Cannot reuse.          %q0 → %q1 → %q2 → %b0
                                        Linear chain. Physically correct.
```

### The Three Dialects

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                         LIFT-CORE                                │
  │   SSA · Types · Operations · Blocks · Regions · Functions        │
  │   Shared foundation — every dialect builds on this               │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
  ┌────────────▼───────────┐    ┌────────────▼───────────┐
  │      LIFT-TENSOR       │    │     LIFT-QUANTUM        │
  │      AI dialect        │    │     QC dialect          │
  │                        │    │                         │
  │  Tensors, shapes       │    │  Qubits (linear types)  │
  │  Auto-diff, gradients  │    │  Gates + noise attrs    │
  │  Attention, KV Cache   │    │  Layout mapping         │
  │  MoE, quantisation     │    │  Hamiltonians, QEC      │
  │  Parallelism strategy  │    │  Error mitigation       │
  └────────────┬───────────┘    └────────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────────────┐
  │                        LIFT-HYBRID                               │
  │   Classical ↔ Quantum data encoding                              │
  │   Parameterised quantum circuits (VQC, QNN)                      │
  │   Joint classical+quantum gradient computation                   │
  │   GPU-side + QPU-side co-execution orchestration                 │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 5. Architecture

```
  ╔════════════════════════════════════════════════════════════════════╗
  ║                        LIFT FRAMEWORK                              ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  USER LAYER                                                        ║
  ║  .lif source  │  .lith config  │  lift(1) CLI  │  Python API      ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  FRONTEND                                                          ║
  ║  Lexer → Parser → AST → Type Checker → SSA Builder                ║
  ║  Importers:  PyTorch FX  │  ONNX  │  Qiskit  │  OpenQASM 3       ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  DIALECT LAYER  (Twin IR)                                          ║
  ║  LIFT-CORE  │  LIFT-TENSOR  │  LIFT-QUANTUM  │  LIFT-HYBRID       ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  SIMULATION + PREDICTION ENGINE                                    ║
  ║  Shape inference  │  FLOP count  │  Noise simulation               ║
  ║  GNN perf predict │  Fidelity    │  Energy budget  │  Carbon       ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  OPTIMISATION PASS PIPELINE                                        ║
  ║  AI:      TensorFusion · FlashAttention · KVCache · INT8/FP8       ║
  ║  Quantum: GateCancellation · SabreLayout · ZNE · QEC               ║
  ║  Hybrid:  HybridFusion · ParameterShift · EncodingOpt              ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  BACKEND LAYER                                                     ║
  ║  CUDA (PTX)  │  OpenQASM 3  │  LLVM IR  │  XLA / StableHLO        ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║  HARDWARE                                                          ║
  ║  H100 · A100 · MI300  │  IBM Kyoto · Rigetti · IonQ  │  TPU       ║
  ╚════════════════════════════════════════════════════════════════════╝
```

### Workspace Layout

```
  lift/
  ├── crates/
  │   ├── lift-core/        SSA IR, types, ops  (zero external deps)
  │   ├── lift-ast/         .lif lexer, parser, AST
  │   ├── lift-tensor/      AI dialect
  │   ├── lift-quantum/     Quantum dialect
  │   ├── lift-hybrid/      Fusion dialect
  │   ├── lift-sim/         Static analysis + quantum simulator
  │   ├── lift-predict/     GNN performance prediction engine
  │   ├── lift-opt/         Pass manager + all optimisation passes
  │   ├── lift-import/      PyTorch FX · ONNX · Qiskit · OpenQASM3
  │   ├── lift-export/      CUDA · OpenQASM3 · LLVM · XLA
  │   ├── lift-config/      .lith configuration language parser
  │   ├── lift-python/      Python bindings (PyO3 / Maturin)
  │   └── lift-cli/         lift(1) command-line interface
  ├── examples/             .lif example programmes
  ├── tests/                integration + regression (5000+ cases)
  └── benches/              benchmark suite
```

---

## 6. The Four Pillars

### Pillar 1 — SIMULATE

Static analysis before any hardware is touched:

```
  .lif module
      │
      ▼  Shape propagation → infer output shapes, catch mismatches early
      │
      ▼  FLOP counting     → per-operation and per-module totals
      │
      ▼  Memory liveness   → peak memory, buffer reuse opportunities
      │
      ▼  Noise analysis    → T1/T2 decoherence risk, expected fidelity
      │
      ▼  Energy model      → joules per op × count + infrastructure overhead
      │
      ▼  SIMULATION REPORT
         ──────────────────────────────────────────────────────────
         AI:      4.7 TFLOPS  ·  12.4 GB peak  ·  1 847 req/s est.
         Quantum: depth=24    ·  87 gates       ·  fidelity=97.3%
         Energy:  0.003 kWh   ·  1.05 gCO₂  (us-east-1 grid)
```

### Pillar 2 — PREDICT

A trained GNN model predicts performance before hardware executes. Budget violations stop compilation with actionable errors:

```
  BUDGET SATISFIED                      BUDGET VIOLATED
  ──────────────────────────            ────────────────────────────────
  Latency:  47ms   ✓ (max 100ms)        ERROR: latency 147ms > 100ms
  Fidelity: 99.1%  ✓ (min 95%)
  Memory:   31 GB  ✓ (max 40 GB)        Suggestions:
  Energy:   0.003  ✓ (max 0.01 kWh)       1. flash-attention → −62ms ✓
                                           2. seq_len 2048→1024 → −80ms ✓
  ✓ Proceed to optimisation.               3. INT8 quant → −44ms  ✗ still over
```

**GNN architecture:** 6 message-passing layers, hidden dim 256, trained on 100K+ (IR graph, hardware spec, measured latency) triples. Falls back to an analytical roofline model when confidence < 0.70.

### Pillar 3 — OPTIMISE

All passes are **semantics-preserving by construction** and validated against 5 000+ reference programmes before release.

```
  AI PASSES
  ────────────────────────────────────────────────────────────────────
  tensor-fusion        Declarative pattern matching O(V+E×P), not
                       Ullmann O(n!). Fuses MatMul+Bias+ReLU etc.
                       Gain: 30–50% less memory bandwidth.

  flash-attention      Replace O(n²) attention with tiled O(n).
                       Triggered when seq_len > 512 on GPU target.
                       Gain: 10–20× on long sequences.

  kv-cache             Pre-allocate key-value memory for LLM inference.
                       Gain: 100× latency reduction (incremental decode).

  quantization         INT8/FP8. Dynamic or static calibration.
                       Gain: 4× model size reduction, 2–4× throughput.

  moe-routing          Expert dispatch for Mixture-of-Experts models.
                       Gain: linear scaling toward T-parameter scale.

  QUANTUM PASSES
  ────────────────────────────────────────────────────────────────────
  gate-cancellation    H·H=I, X·X=I, Rz(a)·Rz(b)=Rz(a+b).
                       Commutation table for safe reordering.
                       Gain: 15–40% depth reduction.

  layout-mapping       SABRE routing (noise-aware variant available).
                       Minimises SWAP insertions on physical topology.

  zne-mitigation       Gate folding (1×, 2×, 3× noise) + Richardson
                       extrapolation. Auto-order by circuit depth.
                       Gain: 5–20× fidelity improvement.

  HYBRID PASSES
  ────────────────────────────────────────────────────────────────────
  hybrid-fusion        Fuse classical post-processing with measurement.
                       Eliminates GPU ↔ QPU round-trips.

  parameter-shift      Expand joint gradients into 2P circuit evaluations.
                       Enables true end-to-end training.
```

### Pillar 4 — COMPILE

```
  OPTIMISED IR
   │
   ├──► CUDA backend      Tensor Core kernels · memory-coalesced ops
   │                      NCCL multi-GPU · CUDA graph capture
   │
   ├──► OpenQASM 3        Gate decomposition to hardware-native basis
   │                      IBM: {RZ, SX, X, CX}  Rigetti: {RZ, RX, CZ}
   │                      Pulse schedule generation for IBM
   │
   ├──► LLVM backend      AVX-512 SIMD · OpenMP multi-core
   │                      Native binary for CPU inference
   │
   └──► Hybrid runner     GPU + QPU orchestration · sync · data transfer
                          Self-contained .lift_bundle for deployment
```

---

## 7. The .lif Language

```lif
// File: qnn_classifier.lif
// Hybrid QNN: classical encoder → 4-qubit quantum layer → output head

#dialect tensor

module @encoder {
    func @encode(%img: tensor<1x784xf32>) -> tensor<1x4xf32> {
        %h0  = "tensor.linear"(%img, %W1, %b1)
               : (tensor<1x784xf32>, tensor<784x64xf32>, tensor<64xf32>)
               -> tensor<1x64xf32>
        %h1  = "tensor.relu"(%h0) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %out = "tensor.linear"(%h1, %W2, %b2)
               : (tensor<1x64xf32>, tensor<64x4xf32>, tensor<4xf32>)
               -> tensor<1x4xf32>
        return %out
    }
}

#dialect quantum

module @q_layer {
    // Qubits are LINEAR: each used exactly once — compiler enforces this
    func @forward(%feat: tensor<4xf32>, %params: tensor<8xf32>)
                  -> (qubit, qubit, qubit, qubit) {
        %q0 = "quantum.init"() : () -> qubit
        %q1 = "quantum.init"() : () -> qubit
        %q2 = "quantum.init"() : () -> qubit
        %q3 = "quantum.init"() : () -> qubit

        // Angle encoding: feature[i] becomes a rotation angle
        %q0 = "quantum.ry"(%q0, %feat[0]) : (qubit, f32) -> qubit
        %q1 = "quantum.ry"(%q1, %feat[1]) : (qubit, f32) -> qubit
        %q2 = "quantum.ry"(%q2, %feat[2]) : (qubit, f32) -> qubit
        %q3 = "quantum.ry"(%q3, %feat[3]) : (qubit, f32) -> qubit

        // Entangling layer (creates quantum correlations)
        %q0, %q1 = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)
        %q2, %q3 = "quantum.cx"(%q2, %q3) : (qubit, qubit) -> (qubit, qubit)

        // Trainable rotation layer
        %q0 = "quantum.rz"(%q0, %params[0]) : (qubit, f32) -> qubit
        %q1 = "quantum.rz"(%q1, %params[1]) : (qubit, f32) -> qubit
        %q2 = "quantum.rz"(%q2, %params[2]) : (qubit, f32) -> qubit
        %q3 = "quantum.rz"(%q3, %params[3]) : (qubit, f32) -> qubit

        return %q0, %q1, %q2, %q3
    }
}

#dialect hybrid

module @classifier {
    func @classify(%image: tensor<1x784xf32>) -> tensor<1x10xf32> {
        // Step 1: classical feature extraction
        %feat = "tensor.call"(@encoder::@encode, %image)
                : (tensor<1x784xf32>) -> tensor<1x4xf32>

        // Step 2: quantum layer (encode + parameterised circuit)
        %q0, %q1, %q2, %q3 = "hybrid.angle_encode_forward"(
                @q_layer::@forward, %feat, %params)
                : (tensor<1x4xf32>, tensor<8xf32>)
                -> (qubit, qubit, qubit, qubit)

        // Step 3: measurement — consumes all qubits (linear type enforced)
        %b0 = "quantum.measure"(%q0) : (qubit) -> bit
        %b1 = "quantum.measure"(%q1) : (qubit) -> bit
        %b2 = "quantum.measure"(%q2) : (qubit) -> bit
        %b3 = "quantum.measure"(%q3) : (qubit) -> bit

        // Step 4: classical output head
        %bits   = "tensor.stack"(%b0, %b1, %b2, %b3)
                  : (bit, bit, bit, bit) -> tensor<4xi1>
        %logits = "tensor.linear"(%bits, %Wout, %bout)
                  : (tensor<4xi1>, tensor<4x10xf32>, tensor<10xf32>)
                  -> tensor<1x10xf32>
        return %logits
    }
}
```

---

## 8. The .lith Configuration

One file replaces 8+ separate config files:

```lith
// File: project.lith — one file controls the entire pipeline

project {
    name        = "hybrid-qnn-classifier"
    version     = "1.0.0"
    description = "QNN classifier: classical encoder + 4-qubit layer"
}

dialects {
    tensor  = "1.0.0"
    quantum = "1.0.0"
    hybrid  = "1.0.0"
}

compilation {
    target {
        type = "hybrid"

        gpu {
            backend         = "cuda"
            arch            = "sm_90"        // H100
            memory_limit_gb = 80
            tensor_cores    = true
        }

        qpu {
            provider           = "ibm"
            backend_name       = "ibm_kyoto"
            shots              = 4096
            optimization_level = 3

            error_mitigation {
                readout_error        = true   // matrix inversion
                dynamical_decoupling = true   // XY-4 sequences
                zero_noise_extrap    = true   // gate folding + Richardson
            }
        }
    }
    mode = "release"
}

optimization {
    pipeline = [
        "canonicalize", "constant-folding",
        "tensor-fusion", "quantization",
        "gate-cancellation", "rotation-merging",
        "layout-mapping", "zne-mitigation",
        "hybrid-fusion", "parameter-shift"
    ]
    passes {
        quantization   { precision = "int8"  calibration = "dynamic" }
        layout-mapping { algorithm = "sabre-noise-aware"  rounds = 3 }
        zne-mitigation { noise_factors = [1, 2, 3]  extrapolation = "richardson" }
    }
}

prediction {
    budget {
        max_latency_ms  = 200
        min_fidelity    = 0.92
        max_memory_gb   = 40
        max_energy_kwh  = 0.01
    }
}

metrics {
    collect = ["latency_ms", "fidelity", "memory_gb", "energy_kwh", "co2_grams"]
}
```

---

## 9. Current Status

| Component | Status | What works today | Next milestone |
|-----------|--------|-----------------|----------------|
| `lift-core` | ✅ Alpha | SSA IR, types, ops, verifier, printer | Incremental compilation |
| `lift-ast` | ✅ Alpha | Lexer, parser, error recovery | Error message quality |
| `lift-tensor` | 🚧 Active | MatMul, Add, ReLU, Conv2D, basic passes | Attention, KV Cache, quantisation |
| `lift-quantum` | 📐 Design | Type system designed, gate enum ready | Gate ops, noise model implementation |
| `lift-hybrid` | 📐 Design | Architecture finalised | All operations |
| `lift-sim` | 🚧 Active | Shape propagation, FLOP counting | QC state vector sim, GNN predictor |
| `lift-predict` | 📐 Design | Architecture designed, data format defined | GNN training pipeline |
| `lift-opt` | 🚧 Active | Pass manager, constant folding, DCE | Fusion pass, quantum passes |
| `lift-import` | 🚧 Active | PyTorch FX ~80%, OpenQASM3 ~60% | ONNX complete, Qiskit from scratch |
| `lift-export` | 🚧 Active | LLVM ~70%, OpenQASM3 ~40% | CUDA PTX backend |
| `lift-config` | 🚧 Active | Core .lith syntax ~60% | Validation, config inheritance |
| `lift-python` | 📐 Design | PyO3 scaffold — not functional | Full Python API |
| `lift-cli` | 🚧 Active | `lift verify`, `lift analyse`, `lift print` | compile, simulate, predict |

**Legend:** ✅ Alpha-stable · 🚧 Active development · 📐 Design only (not yet coded)

---

## 10. Getting Started

### Prerequisites

```bash
# Rust 1.78 or newer (mandatory)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Optional: CUDA toolkit for GPU backend
# https://developer.nvidia.com/cuda-downloads

# Optional: Python 3.10+ for Python bindings
pip install maturin
```

### Build from Source

```bash
git clone https://github.com/lift-framework/lift
cd lift
cargo build --release
```

### What You Can Do Today

```bash
# Write a simple tensor programme
cat > hello.lif << 'EOF'
#dialect tensor
module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        %out = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %out
    }
}
EOF

lift verify  hello.lif    # check IR well-formedness
lift analyse hello.lif    # FLOPs, shapes, memory estimate
lift print   hello.lif    # pretty-print the IR
```

---

## 11. Roadmap

**24 months to v1.0 — honest, not optimistic.**

```
  Phase 0  LIFT-CORE          Weeks  1–8    ██████████░░░░░░  DONE
  Phase 1  LIFT-TENSOR        Weeks  5–18   ░░░██████████░░░  ACTIVE
  Phase 2a Basic Quantum      Weeks 15–24   ░░░░░░░░░░░░░░░░  DESIGN
  Phase 2b Advanced Quantum   Weeks 22–36   ░░░░░░░░░░░░░░░░  FUTURE
  Phase 3  LIFT-HYBRID        Weeks 28–42   ░░░░░░░░░░░░░░░░  FUTURE
  Phase 4  SIM + PREDICT      Weeks 32–46   ░░░░░░░░░░░░░░░░  FUTURE
  Phase 5  BACKENDS + IMPORT  Weeks 38–56   ░░░░░░░░░░░░░░░░  FUTURE
  Phase 6  TOOLING            Weeks 52–62   ░░░░░░░░░░░░░░░░  FUTURE
  Phase 7  v1.0 PUBLIC        Week  ~96     TARGET: Q4 2026
```

| Milestone | Target | Criteria |
|-----------|--------|---------|
| Phase 1 complete | Month 5 | LLaMA 7B compiles to LLVM, correct output |
| Phase 2b complete | Month 9 | VQE H₂ runs on IBM Kyoto, correct energy |
| Phase 3 complete | Month 12 | QNN MNIST trains end-to-end with joint gradients |
| Phase 5 complete | Month 18 | LLaMA 7B on H100 within 10% of TensorRT |
| v1.0 release | Month 24 | arXiv preprint + benchmarks published |

---

## 12. Contributing

We need help. Here is where contributions have the most impact:

| Area | Difficulty | What to build |
|------|-----------|---------------|
| FlashAttention pass | Hard | Pattern match + replace in LIFT-TENSOR |
| State vector simulator | Medium | QC simulator CPU + GPU |
| PyTorch FX importer | Medium | Complete remaining 20% |
| Qiskit importer | Medium | Build from scratch |
| CUDA PTX backend | Hard | PTX generation for tensor ops |
| .lith parser | Medium | Validation + inheritance |
| API documentation | Easy | rustdoc for all public items |
| Tutorials | Easy | Getting started guides |

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, PR process, and onboarding.

---

## 13. Known Limitations

### Hard Limits Today

- No quantum hardware backend (design only).
- No GPU code generation (CUDA planned, LLVM partial).
- Python bindings are not functional.
- GNN performance predictor does not exist yet.
- Energy and carbon modelling not implemented.

### Open Design Problems (Documented, Not Blocking)

| Problem | Status | Plan |
|---------|--------|------|
| Linear types in conditional branches | Solution drafted | Region-based analysis in Phase 2a |
| Noise composition after gate fusion | Partially solved | Depolarising approx v1.0, Kraus v1.1 |
| GNN predictor generalisation | Architecture designed | Ensemble with analytical fallback |

---

## 14. Why This Matters

The AI+Quantum convergence is not a hypothetical future — IBM targets 1 000+ qubits by 2026. Hybrid variational algorithms (VQE, QAOA, QNN) are moving from academic prototypes to industrial applications. AI models are reaching scales where new computing paradigms are needed.

**The unified toolchain does not exist yet.** Two ecosystems are forming independently. If they ossify before being bridged, the integration cost grows exponentially. LIFT's bet: build the bridge now, with the correct foundations, before the window closes.

We are not claiming a finished product. We are claiming a correct architecture, an honest implementation plan, and the conviction that this problem is worth solving well.

---

```bibtex
@software{lift2025,
  title  = {LIFT: Language for Intelligent Frameworks and Technologies},
  author = {Martial-Christian and Contributors},
  year   = {2025},
  url    = {https://github.com/lift-framework/lift},
  note   = {Unified IR for AI and Quantum Computing. Research Alpha.}
}
```

**License:** MIT — see [LICENSE](LICENSE).

---

<div align="center">

*LIFT — Because the future of computation is both intelligent and quantum,*
*and it deserves a unified foundation.*

</div>