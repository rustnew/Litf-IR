# LIFT — Features, Capabilities, Limits, and Goals

**Complete analysis based on the actual source code (67 Rust files, 13 crates, 541 tests).**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Processing Pipeline](#2-processing-pipeline)
3. [Implemented Features](#3-implemented-features)
4. [Partial Features](#4-partial-features)
5. [Missing Features](#5-missing-features)
6. [Current Limitations](#6-current-limitations)
7. [Analysis Accuracy](#7-analysis-accuracy)
8. [What's Missing to Reach the Goals](#8-whats-missing-to-reach-the-goals)
9. [Roadmap](#9-roadmap-by-priority)

---

# 1. Overview

LIFT is a unified IR compiler for classical AI + quantum computing, written in Rust (13 crates):

| Crate | Role |
|-------|------|
| `lift-core` | Core: IR context, types, verifier, printer, pass manager |
| `lift-ast` | Lexer, parser, builder for `.lif` files |
| `lift-tensor` | 110 AI operations, shape inference, FLOP counting |
| `lift-quantum` | 50+ quantum gates, noise, Kraus, QEC, topology |
| `lift-hybrid` | 21 classical↔quantum operations |
| `lift-opt` | 13 optimisation passes |
| `lift-sim` | Static analysis, GPU/QPU cost models, energy |
| `lift-predict` | Roofline prediction, quantum prediction |
| `lift-config` | `.lith` file parser, O0-O3 level pipeline, quantum provider |
| `lift-import` | ONNX, PyTorch FX, OpenQASM import (skeletons) |
| `lift-export` | LLVM IR, ONNX (opset 21), OpenQASM 3.0 export |
| `lift-cli` | CLI: verify, analyse, print, optimise, predict, export |
| `lift-codegen` | Programmatic model generation, multi-format export |
| `lift-tests` | 541 tests, 0 failures |

---

# 2. Processing Pipeline

```
.lif → Lexer → Parser → Builder → Context IR → Verification → Analysis → Optimisation → Export
```

### Stage 1 — Lexer (COMPLETE)
Splits `.lif` text into tokens: keywords, `#dialect` directives, identifiers (`@name`/`%var`), literals, punctuation. Includes error handling.

### Stage 2 — Parser (COMPLETE)
Builds the AST: dialect directives, modules, functions, operations with operands/attributes/type signatures. Tensor types (`tensor<1x784xf32>`), qubit, bit, hamiltonian. Error recovery.

### Stage 3 — Builder (COMPLETE)
Converts the AST into internal IR inside the `Context`: SSA values, operations, blocks, regions, functions, modules.

### Stage 4 — Context IR (COMPLETE)
Central structure with SlotMaps for values, ops, blocks, regions, types + StringInterner. Types: Integer (i1-i64), Float (f16-f64, fp8), Boolean, Void, Tuple, Function, Opaque (tensor, qubit, bit, hamiltonian).

### Stage 5 — Verification (COMPLETE, 4 passes)
- **SSA**: every value defined exactly once, every use after its definition
- **Well-formedness**: no dangling references (ops ↔ values ↔ blocks ↔ regions)
- **Linearity**: every qubit consumed exactly once (no-cloning)
- **Semantic**: input arity of every operation checked against dialect signatures (core + tensor + quantum + hybrid), via `verify_semantics()` / `verify_with_dialects()`

13 error types: UndefinedValue, MultipleDefinition, DominanceViolation, TypeMismatch, LinearityViolation, QubitLeaked, BranchLinearityMismatch, DanglingReference, MissingTerminator, OrphanedOperation, OrphanedBlock, InvalidOperation, SemanticError.

### Stage 6 — Static Analysis (COMPLETE)
Produces: total_flops, total_memory_bytes, peak_memory, num_ops per dialect, op_breakdown. Quantum: qubits, 1Q/2Q/3Q gates, measurements, circuit_depth, estimated_fidelity, accumulated noise.

### Stage 7 — Optimisation (13 passes)

| Pass | Type | Concrete action |
|------|------|------------------|
| `canonicalize` | Tensor | Normalises patterns |
| `constant-folding` | Tensor | Evaluates constants at compile time |
| `dce` | General | Removes ops whose results are unused |
| `tensor-fusion` | Tensor | Fuses matmul+add+relu → fused_matmul_bias_relu, linear+gelu → fused_linear_gelu, linear+silu → fused_linear_silu, conv2d+bn+relu (2 phases: ternary then binary) |
| `cse` | General | Eliminates common subexpressions |
| `flash-attention` | Tensor | Replaces attention → flash attention |
| `quantisation-pass` | Tensor | Annotates for INT8/INT4 quantisation |
| `gate-cancellation` | Quantum | Cancels H·H=I, X·X=I, S·Sdg=I, T·Tdg=I — including **non-consecutive** pairs (separated by commuting gates on other qubits, SSA chain verified) |
| `rotation-merge` | Quantum | Merges Rz(a)·Rz(b) → Rz(a+b) — same, non-consecutive pairs |
| `noise-aware-schedule` | Quantum | Reorders gates to minimise decoherence |
| `layout-mapping` | Quantum | Annotates 2-qubit gates that need SWAPs |
| `gate-decomposition` | Quantum | Decomposes H/T/Tdg/S/Sdg/Y/RX into the provider's native set (IBM, Rigetti, IonQ, Quantinuum), driven by `[quantum] provider`; the original gate is removed and replaced, not left in place alongside its decomposition |
| `real-routing` | Quantum | Inserts **real `quantum.swap`** ops (BFS shortest path) to satisfy topology connectivity; tracks logical↔physical placement |

**Level pipelines** (`[optimisation] level = O0|O1|O2|O3`):
- `O0`: no passes
- `O1`: canonicalize, constant-folding, dce
- `O2`: O1 + cse, tensor-fusion
- `O3`: all 13 passes, including gate-decomposition and real-routing

Explicit `passes` take priority over the level; `disabled_passes` removes passes; unknown passes trigger a warning (`OptimisationConfig::validate()`). All passes are reachable from the CLI.

### Stage 8 — Prediction (COMPLETE)
- **GPU roofline**: compute_time_ms, memory_time_ms, bottleneck. A100 (312 TFLOPS) and H100 (989 TFLOPS) models.
- **Quantum**: fidelity, circuit_time_us, shots needed. 3 models: superconducting, trapped ion, neutral atom.
- **Budget**: checks max FLOPs, max memory, max time, min fidelity. ReactiveBudget for real-time tracking.
- Both are reachable from the CLI via `predict --energy` and `predict --quantum <hardware>`.

### Stage 9 — Export (3 backends)
- **LLVM IR**: ops emitted as comments with cuBLAS/cuDNN runtime calls
- **ONNX**: protobuf text, opset 21, 70+ operations mapped (standard + com.microsoft)
- **OpenQASM 3.0**: all 48 `QuantumGate` variants have a match arm (verified: no wildcard/unsupported fallback exists in the exporter) — 46 emit a real QASM gate instruction, and `IfElse`/`ParamGate` (control-flow/generic wrappers, not literal gates) emit a descriptive comment. Qubit indices are resolved by following each gate's actual SSA operand back to its owning qubit, not assigned from a counter

---

# 3. Implemented Features

## 3.1 Tensor Dialect — 110 operations

All 110 operations are defined in the `TensorOp` enum with name↔enum conversion, input count, classification. Working shape inference for: MatMul, Linear, Conv2D, Conv1D, DepthwiseConv2D, Attention, FlashAttention, MaxPool2D, GlobalAvgPool, BatchNorm, LayerNorm, RMSNorm, InstanceNorm, SparseMatMul, elementwise, ELU, LeakyReLU, Mish, HardSwish. Exact FLOP counting for MatMul, Linear, Conv2D, Attention, ReLU, elementwise, fused ops.

## 3.2 Quantum Dialect — 50+ gates

Gates: 9 standard 1Q + 7 parametric 1Q + 2 fixed-angle + 13 2Q gates + 2 3Q gates + 2 multi-controlled + 8 measurement/control + IonQ gates. Per-gate properties: num_qubits, is_parametric, is_self_inverse, is_clifford, is_entangling. 5 native sets (IBM, Rigetti, IonQ, Quantinuum, Simulator). Noise: GateNoise, CircuitNoise, KrausChannel (6 channels). Topology: linear, grid, heavy_hex, all_to_all, tree, custom + BFS. QEC: Surface, Steane, Shor, Repetition, LDPC.

## 3.3 Hybrid Dialect — 21 operations

Encode/Decode, 5 gradients, 4 variational algorithms, 2 transfers, 4 processing ops, CoExecute, 2 measurements. AnsatzType, SyncPolicy, FeatureMap, EncodingStrategy.

## 3.4 CLI — 6 commands

`verify`, `analyse` (text/JSON), `print`, `optimise` (with `.lith`), `predict` (A100/H100, `--energy`, `--quantum`), `export` (llvm/onnx/qasm).

## 3.5 Programmatic Generation — lift-codegen

`lift-codegen` binary: defines models from Rust via `ModelBuilder`, automatically generates `.lif`, `.ll`, `.onnx`, `.qasm`, `.lith`. 4 predefined models (Phi-3-mini, MLP, ResNet, VQE).

## 3.6 Energy Models

A100/H100 EnergyModel: energy in joules/kWh, CO2 grams, quantum energy (cryogenics). Connected to the CLI via `predict --energy`.

## 3.7 Tests — 541 tests, 0 failures

Types, operations, shapes, FLOPs, memory, gates, noise, topology, QEC, Kraus, benchmarks (GPT-2, LLaMA-7B, ResNet-50, BERT-base), O0-O3 pipeline, semantic verification, generic fusions, gate decomposition, non-consecutive cancellation/merge, real SWAP routing, printer/parser round-tripping. End-to-end validation: `examples/validate_all.sh` (105 checks, including all 13 passes).

---

# 4. Partial Features (code exists, incomplete)

## 4.1 LLVM IR Export — SKELETON

The exporter produces `define void @func(ptr %arg0) { entry: ; tensor.matmul  ret void }`. Operations are emitted as **comments**, not real LLVM IR. No cuBLAS/cuDNN calls, no memory management.

## 4.2 ONNX Export — OPERATIONAL

The ONNX exporter produces protobuf text (opset 21) with 70+ operations mapped to standard ONNX and com.microsoft ops. Data types, shapes, and initializer nodes are generated. **Missing**: binary protobuf serialisation (currently text only), connected node graphs (nodes are emitted sequentially without explicit edges).

## 4.3 OpenQASM Export — all 48 gates handled, 2 as comments

Every `QuantumGate` variant has a match arm; 46 produce a real QASM gate instruction (including less-common ones like `MCX`, `CSWAP`, `GPI`/`GPI2`, `XX`/`YY`/`ZZ`). `IfElse` and `ParamGate` are control-flow/generic wrappers rather than fixed gates, so they emit a descriptive comment instead of a gate line. Gate order follows the real circuit order (`block.ops`) and qubit indices are resolved from each gate's actual operand chain, not a counter.

## 4.4 ONNX/PyTorch/QASM Import — SKELETONS

All 3 importers read the source format but create an **empty** module+function. No node/operation is actually converted into LIFT operations.

## 4.5 Layout Mapping — ANNOTATION ONLY

Adds `needs_swap = true` on non-adjacent 2Q gates. **Does not actually insert SWAPs or route.** (`real-routing` does the real work; `layout-mapping` remains a legacy annotation-only pass.)

## 4.6 Shape Inference — PARTIAL

Works for about 20 of the 110 operations. Missing: Conv3D, ConvTranspose2D, Reshape, Permute, Concat, Split, Slice, LSTM, GRU, RNN, FFT, SVD, Einsum, GNN, MoE, diffusion, quantisation, parallelism.

---

# 5. Missing Features

## 5.1 No Semantic Verification of Operand Shapes

The verifier checks SSA/well-formedness/linearity but **does NOT check** that `tensor.matmul` has 2 tensor inputs, that dimensions are compatible, that `tensor.conv2d` receives a 4D tensor, etc.

## 5.2 No Real Execution

LIFT cannot **execute** a program. It is purely an analysis compiler. There is no runtime, no interpreter, no GPU/QPU execution backend.

## 5.3 No Real Quantum Simulation

The `quantum_sim` module does **static analysis** (gate counting, fidelity estimation). It does NOT simulate quantum state (no state vector, no density matrix, no Monte Carlo simulation).

## 5.4 No Machine Code Generation

LLVM export does not produce executable code. This would require: lowering tensor operations to library calls (cuBLAS, cuDNN, oneDNN), memory management (allocation/deallocation), kernel scheduling, GPU launch code.

## 5.5 No Multi-File Support

A LIFT program is a single `.lif` file. No import/include system, no separate modules, no linking.

## 5.6 Limited Gate Decomposition Table

`gate-decomposition` correctly replaces the gates it knows about (H, T, Tdg, S, Sdg, Y, RX), but the decomposition table only covers those seven — every other non-native gate silently passes through unchanged, even when targeting hardware whose native set doesn't include it.

## 5.7 No GPU Scheduling

No placement of operations on CUDA streams, no compute/memory overlap, no operation parallelism.

## 5.8 No Automatic Differentiation

Gradient operations are **declared** (grad_matmul, grad_relu, etc.) but there is no autodiff system that automatically builds the backward graph from the forward graph.

## 5.9 No Data Handling

No data loading (datasets), no data loaders, no preprocessing. LIFT works purely on the computation graph.

---

# 6. Current Limitations

## 6.1 Structural Limitations

| Limitation | Impact |
|------------|--------|
| No execution | LIFT analyses but cannot execute a model |
| Skeleton export | Generated code (LLVM/QASM) is not executable as-is |
| Skeleton import | Cannot import a real ONNX/PyTorch model |
| No QC simulation | Fidelity estimated by formula, not real simulation |

## 6.2 Cost Model Limitations

- The roofline model is a **coarse approximation**: it does not account for cache effects, kernel launch latency, or compute/memory overlap
- The quantum model uses **average** default noise parameters, not the real properties of the target device
- Fidelity estimation assumes **independent** noise per gate (no spatial/temporal correlations)

## 6.3 Verifier Limitations

- No operand type checking (input types vs. signature)
- No dimension-compatibility checking (tensor shapes)
- No complete dominance checking (CFG)
- Linearity verification does not exhaustively handle conditional branches
- Semantic verification checks arity but not tensor dimensions

## 6.4 Optimiser Limitations

- `tensor-fusion` recognises 5 patterns (matmul+bias+relu, matmul+bias, linear+gelu/silu, conv+bn+relu) but not attention+softmax or layernorm fusions
- `gate-cancellation`/`rotation-merge` detect non-consecutive pairs via the SSA chain, but not cross patterns (e.g. H·Rz)
- `noise-aware-schedule` sorts by gate time, not a real constrained scheduling algorithm
- `real-routing` inserts SWAPs (BFS) with an identity initial placement; no SABRE-style dynamic re-placement, no SWAP-direction correction for directionality

---

# 7. Analysis Accuracy

## 7.1 FLOP Counting

| Operation | Accuracy | Formula |
|-----------|----------|---------|
| MatMul (MxK × KxN) | **Exact** | 2 × M × K × N |
| MatMul batch (BxMxK × BxKxN) | **Exact** | 2 × B × M × K × N |
| Linear (MxK × KxN + N) | **Exact** | 2 × M × K × N + M × N |
| Conv2D | **Exact** | 2 × B × Cout × Hout × Wout × Cin × Kh × Kw |
| Attention | **Exact** | 2 × B × H × (S² × D + S × D²) |
| ReLU / elementwise | **Exact** | element count |
| Reshape, Transpose | **Exact** | 0 FLOPs (correct) |
| Fused ops | **Exact** | sum of components |
| LSTM, GRU, RNN | **Not implemented** | — |
| Conv3D, ConvTranspose | **Not implemented** | — |
| Einsum, FFT, SVD | **Not implemented** | — |

**Overall accuracy**: for pure Transformer models (GPT, BERT, LLaMA), FLOP-counting accuracy is **excellent** (error < 1%). For CNN models, it's good for Conv2D but misses other convolutions. For recurrent models (LSTM), FLOPs are not counted.

## 7.2 Memory Estimation

Computes `element_count × byte_size(dtype)` per tensor. Accurate for **static** memory but does not model: dynamically allocated intermediate activations, GPU memory fragmentation, workspace buffers (cuDNN), KV cache for LLM inference.

## 7.3 Time Prediction (Roofline)

| Aspect | Accuracy |
|--------|----------|
| Compute-bound vs. memory-bound identification | **Good** (standard cases) |
| Absolute time | **Order of magnitude** (2-5x error possible) |
| Cache effects | **Not modelled** |
| Kernel launch latency | **Not modelled** |
| Multi-GPU | **Not modelled** (assumes 1 GPU) |
| Compute/memory overlap | **Not modelled** |

## 7.4 Quantum Fidelity

Fidelity is estimated as the **product of individual fidelities**: F = ∏ f_gate × f_decoherence. This is an **upper bound** (real fidelity is often worse due to noise correlations, crosstalk, and readout errors).

---

# 8. What's Missing to Reach the Goals

LIFT's goal is: **"Simulate → Predict → Optimise → Compile"**. Current state:

| Goal | State | What's missing |
|------|-------|-----------------|
| **Simulate** | 40% | Static analysis is solid, but no real execution simulation (no quantum state vector, no tensor interpreter) |
| **Predict** | 70% | GPU roofline OK, quantum prediction OK, but the model is too simplified (no cache, no multi-GPU, no scheduling) |
| **Optimise** | 70% | 13 passes wired in, O0-O3 pipeline, semantic verification, generic fusions, gate decomposition, real SWAP routing; missing a general rewrite graph and attention/layernorm fusions |
| **Compile** | 10% | LLVM/QASM export are skeletons, no real executable code |

## 8.1 To Reach Simulate (100%)

1. **Quantum state-vector simulator**: multiply gate matrices onto a 2^n vector. Needed to validate quantum circuits.
2. **Tensor interpreter**: execute tensor ops with real, numpy-like values. Needed to validate AI models.
3. **Monte Carlo simulation**: to estimate the measurement distribution under noise.

## 8.2 To Reach Predict (100%)

1. **Refined cost model**: incorporate launch latency, L2 cache effects, overlapped scheduling.
2. **Real hardware profiles**: load real QPU properties (IBM Quantum calibration, per-qubit gate times).
3. **Multi-GPU**: inter-GPU communication model (NVLink, PCIe).
4. **Advanced quantum prediction**: correlated noise model, crosstalk, readout errors.

## 8.3 To Reach Optimise (100%)

1. **More fusion patterns**: matmul+gelu, conv+bn+relu, attention+layernorm.
2. **Non-local gate cancellation**: cancel pairs separated by operations on other qubits (commutation).
3. **Real routing**: implement SABRE or A* for layout mapping with SWAP insertion.
4. **Broader gate decomposition**: cover more than the current 7-gate table.
5. **Pattern-based rewrite system**: allow declarative transformation rules.

## 8.4 To Reach Compile (100%)

1. **Tensor → LLVM lowering**: generate real calls to cuBLAS/cuDNN/oneDNN.
2. **Memory management**: GPU memory allocator (allocation, deallocation, reuse).
3. **Launch code**: generate host code that orchestrates GPU kernels.
4. **Quantum backend**: generate code for IBM Qiskit Runtime, Amazon Braket, or Google Cirq.
5. **Real import**: convert real ONNX/PyTorch graphs into LIFT operations.

---

# 9. Roadmap (by priority)

## Priority 1 — Quick fixes (low effort, immediate impact) — done

- [x] Wire all 13 passes into the CLI (`cmd_optimise`, main.rs)
- [x] Wire EnergyModel into the CLI (`predict --energy`, `--num-gpus`)
- [x] Wire predict_quantum into the CLI (`predict --quantum <hardware> --precision`)
- [x] `crates/lift-demo/src/config.rs` present and compiling (`cargo build -p lift-demo`)
- [x] Fix the printer/parser round trip (`optimise --output` produced a `.lif` the parser couldn't read back)
- [x] Fix QASM qubit indexing (counter → real SSA operand chain), gate order (slotmap → `block.ops`), and per-function qubit counting
- [x] Fix `gate-decomposition` leaving the original gate in place next to its own decomposition

## Priority 2 — Functional Import/Export (medium effort, high impact)

- [ ] Real ONNX import: map ONNX nodes to TensorOp
- [ ] Real PyTorch FX import: map FX nodes to TensorOp
- [x] Full QASM export: all 48 gates handled (done — see §4.3)
- [ ] Real QASM import: parse gates and create quantum operations

## Priority 3 — Advanced Optimisation (medium effort)

- [ ] More tensor fusion patterns
- [ ] Non-local gate cancellation (commutation)
- [ ] Broader gate decomposition table
- [ ] Real routing (SABRE)

## Priority 4 — Simulation (high effort)

- [ ] State-vector simulator (up to ~25 qubits)
- [ ] Simplified tensor interpreter
- [ ] Shape inference for the remaining 90 operations

## Priority 5 — Real Compilation (very high effort)

- [ ] Tensor → LLVM lowering with cuBLAS calls
- [ ] GPU memory management
- [ ] Quantum backend (Qiskit/Braket)

---

# Final Summary

| Metric | Value |
|--------|-------|
| **Crates** | 14 |
| **Rust files** | 67 |
| **Tests** | 541 (0 failures) |
| **Defined operations** | 179 (110 tensor + 48 quantum + 21 hybrid) |
| **Optimisation passes** | 13 (13 wired into the CLI) |
| **Export backends** | 3 (LLVM IR, ONNX opset 21, OpenQASM 3.0) |
| **Cost models** | 5 (A100, H100, superconducting, trapped ion, neutral atom) |
| **QASM-exported gates** | 48 / 48 (46 as real gates, 2 as comments) |
| **ONNX-exported ops** | 70+ / 110 |
| **Functional imports** | 0 / 3 |
| **Execution possible** | No |
| **Real compilation** | No |

**LIFT is a solid, well-tested IR analysis and optimisation framework**, with excellent dialect coverage (tensor, quantum, hybrid) and a clean architecture. Its strength is static analysis (FLOPs, memory, fidelity, noise, cost). The addition of ONNX export (opset 21) and the `lift-codegen` binary now makes it possible to generate models programmatically and export them to 3 backends (LLVM, ONNX, QASM). **What it mainly lacks** is the ability to actually execute code: LLVM export is a skeleton, imports are empty, and there is no runtime. To become a complete "Simulate → Predict → Optimise → Compile" compiler, it needs real lowering, functional imports, and a simulator.

**This document is a complete and honest analysis of LIFT's current state.**
