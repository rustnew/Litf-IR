# LIFT User Manual — Complete Usage Guide

> **LIFT** — *Language for Intelligent Frameworks and Technologies*
> Version 0.2.1

This manual is the definitive reference for every use case of the LIFT compiler framework. It presents **real-world problems**, explains how LIFT solves them, and provides **working code examples** for each scenario.

---

## Table of Contents

1. [What is LIFT and Why Does It Exist?](#1-what-is-lift-and-why-does-it-exist)
2. [Installation and Setup](#2-installation-and-setup)
3. [Core Concepts](#3-core-concepts)
4. [The `.lif` Source Language](#4-the-lif-source-language)
5. [Use Case 1 — Neural Network Optimisation](#5-use-case-1--neural-network-optimisation)
6. [Use Case 2 — Transformer Attention and FlashAttention](#6-use-case-2--transformer-attention-and-flashattention)
7. [Use Case 3 — Quantum Circuit Design and Noise Analysis](#7-use-case-3--quantum-circuit-design-and-noise-analysis)
8. [Use Case 4 — Hybrid Classical-Quantum (VQE)](#8-use-case-4--hybrid-classical-quantum-vqe)
9. [Use Case 5 — Model Import](#9-use-case-5--model-import)
10. [Use Case 6 — Performance Prediction](#10-use-case-6--performance-prediction)
11. [Use Case 7 — Quantised Inference](#11-use-case-7--quantised-inference)
12. [Use Case 8 — Backend Export](#12-use-case-8--backend-export)
13. [Use Case 9 — Energy and Carbon Estimation](#13-use-case-9--energy-and-carbon-estimation)
14. [Use Case 10 — Device Topology and Routing](#14-use-case-10--device-topology-and-routing)
15. [Use Case 11 — Diffusion and GNN Models](#15-use-case-11--diffusion-and-gnn-models)
16. [Use Case 12 — Budget-Constrained Compilation](#16-use-case-12--budget-constrained-compilation)
17. [Use Case 13 — End-to-End Pipelines](#17-use-case-13--end-to-end-pipelines)
18. [Configuration with `.lith` Files](#18-configuration-with-lith-files)
19. [CLI Reference](#19-cli-reference)
20. [Complete API Reference](#20-complete-api-reference)
21. [Troubleshooting](#21-troubleshooting)

---

## 1. What is LIFT and Why Does It Exist?

### 1.1 The Problem

Modern computing faces a fragmentation crisis:

- **AI/ML frameworks** (PyTorch, TensorFlow, ONNX) produce models in incompatible formats with no unified optimisation pipeline.
- **Quantum computing** (Qiskit, Cirq, OpenQASM) uses entirely separate toolchains with no connection to classical compilation.
- **Hybrid algorithms** (VQE, QAOA, Quantum ML) require ad-hoc glue code between classical and quantum systems.
- **Performance analysis** is fragmented — different tools for GPU profiling, quantum fidelity, and cost modelling.

### 1.2 How LIFT Solves It

LIFT provides a **single SSA-based intermediate representation** spanning three dialects:

| Dialect | Domain | Operations |
|---------|--------|------------|
| **tensor** | AI/ML | 107 ops: arithmetic, attention, convolution, normalisation, quantisation, GNN, diffusion |
| **quantum** | Quantum computing | 46+ gates: Pauli, Clifford, parametric, multi-qubit; noise models, QEC, topology |
| **hybrid** | Classical-quantum | 21 ops: encoding, gradient methods, variational algorithms, GPU↔QPU transfer |

The unified pipeline: **import → verify → analyse → optimise → predict → export**.

### 1.3 Architecture

```
                    ┌──────────┐
                    │ lift-cli │  ← User interface
                    └────┬─────┘
           ┌─────────────┼─────────────┐
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │ lift-import │ │lift-opt │ │ lift-export │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │  lift-ast   │ │lift-sim │ │lift-predict │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
    ┌──────┴─────────────┴─────────────┴──────┐
    │              lift-core                    │
    ├──────────┬──────────┬───────────────────┤
    │lift-tensor│lift-quantum│  lift-hybrid    │
    └──────────┴──────────┴───────────────────┘
```

---

## 2. Installation and Setup

### 2.1 Prerequisites

- **Rust 1.80+** — install via [rustup](https://rustup.rs/)

### 2.2 Build

```bash
git clone https://github.com/rustnew/Lift.git
cd Lift
cargo build --release
cargo test --workspace   # 505 tests, all pass
```

### 2.3 Use as a Library

```toml
[dependencies]
lift-core    = { path = "crates/lift-core" }
lift-ast     = { path = "crates/lift-ast" }
lift-tensor  = { path = "crates/lift-tensor" }
lift-quantum = { path = "crates/lift-quantum" }
lift-hybrid  = { path = "crates/lift-hybrid" }
lift-opt     = { path = "crates/lift-opt" }
lift-sim     = { path = "crates/lift-sim" }
lift-predict = { path = "crates/lift-predict" }
lift-import  = { path = "crates/lift-import" }
lift-export  = { path = "crates/lift-export" }
lift-config  = { path = "crates/lift-config" }
```

---

## 3. Core Concepts

### 3.1 SSA IR

LIFT uses **Static Single Assignment** — every value is defined exactly once:

```
%h1 = "tensor.matmul"(%x, %w) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
%h2 = "tensor.relu"(%h1) : (tensor<1x256xf32>) -> tensor<1x256xf32>
```

### 3.2 The Context

The `Context` is the central data structure holding all IR elements:

```rust
use lift_core::{Context, Attributes, Location};
use lift_core::types::{Dimension, DataType, MemoryLayout};

let mut ctx = Context::new();

// Create types
let tensor_ty = ctx.make_tensor_type(
    vec![Dimension::Constant(1), Dimension::Constant(784)],
    DataType::FP32, MemoryLayout::Contiguous,
);
let qubit_ty = ctx.make_qubit_type();

// Create block, add arguments
let block = ctx.create_block();
let x = ctx.create_block_arg(block, tensor_ty);

// Create operation
let (op, results) = ctx.create_op(
    "tensor.relu", "tensor",
    vec![x], vec![tensor_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, op);
```

### 3.3 Linear Qubit Types

**Problem:** Qubits cannot be copied (no-cloning theorem). Classical IRs allow reuse, violating physics.

**Solution:** LIFT enforces linear types for qubits — each consumed exactly once:

```rust
let mut ctx = Context::new();
let qubit_ty = ctx.make_qubit_type();
let block = ctx.create_block();
let q0 = ctx.create_block_arg(block, qubit_ty);

// First use — OK
let (op1, _) = ctx.create_op("quantum.x", "quantum",
    vec![q0], vec![qubit_ty], Attributes::new(), Location::unknown());
ctx.add_op_to_block(block, op1);

// Second use of same q0 — LINEARITY VIOLATION
let (op2, _) = ctx.create_op("quantum.h", "quantum",
    vec![q0], vec![qubit_ty], Attributes::new(), Location::unknown());
ctx.add_op_to_block(block, op2);

let result = lift_core::verifier::verify(&ctx);
assert!(result.is_err());  // VerifyError::LinearityViolation
```

### 3.4 Verification

The verifier checks **SSA**, **dominance**, and **linearity**:

```rust
use lift_core::verifier;

match verifier::verify(&ctx) {
    Ok(()) => println!("IR is valid"),
    Err(errors) => {
        for err in &errors {
            eprintln!("Error: {}", err);
        }
    }
}
```

### 3.5 Printing the IR

```rust
use lift_core::printer::print_ir;
let output = print_ir(&ctx);
println!("{}", output);
```

---

## 4. The `.lif` Source Language

### 4.1 Syntax

```
#dialect tensor

module @name {
    func @function(%arg0: type0, %arg1: type1) -> return_type {
        %result = "dialect.op"(%arg0, %arg1) {attr = value}
            : (type0, type1) -> return_type
        return %result
    }
}
```

### 4.2 Types

| Type | Syntax | Example |
|------|--------|---------|
| Tensor | `tensor<shape x dtype>` | `tensor<1x784xf32>` |
| Qubit | `qubit` | `qubit` |
| Classical bit | `bit` | `bit` |
| Scalar | `f32`, `f64`, `i32`, `i64` | `f32` |

### 4.3 Parsing Programmatically

```rust
use lift_ast::{Lexer, Parser, IrBuilder};
use lift_core::Context;

let source = std::fs::read_to_string("examples/tensor_mlp.lif").unwrap();
let tokens = Lexer::new(&source).tokenize().to_vec();
let program = Parser::new(tokens).parse().unwrap();

let mut ctx = Context::new();
IrBuilder::new().build_program(&mut ctx, &program).unwrap();
lift_core::verifier::verify(&ctx).unwrap();
```

---

## 5. Use Case 1 — Neural Network Optimisation

### 5.1 Problem

You have a Multi-Layer Perceptron (MLP) and want to:
1. Represent it as LIFT IR
2. Verify correctness
3. Fuse MatMul + Bias + ReLU into a single kernel
4. Measure FLOPs and memory

### 5.2 The MLP in `.lif`

File: `examples/tensor_mlp.lif`

```
#dialect tensor

module @mlp {
    func @forward(%x: tensor<1x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>,
                  %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> tensor<1x10xf32> {
        %h1 = "tensor.matmul"(%x, %w1) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
        %h2 = "tensor.add"(%h1, %b1) : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
        %h3 = "tensor.relu"(%h2) : (tensor<1x256xf32>) -> tensor<1x256xf32>
        %h4 = "tensor.matmul"(%h3, %w2) : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
        %h5 = "tensor.add"(%h4, %b2) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
        %out = "tensor.softmax"(%h5) : (tensor<1x10xf32>) -> tensor<1x10xf32>
        return %out
    }
}
```

### 5.3 Build the IR Programmatically

```rust
use lift_core::{Context, Attributes, Location};
use lift_core::types::{Dimension, DataType, MemoryLayout};

let mut ctx = Context::new();

let input_ty = ctx.make_tensor_type(
    vec![Dimension::Constant(1), Dimension::Constant(784)],
    DataType::FP32, MemoryLayout::Contiguous,
);
let w1_ty = ctx.make_tensor_type(
    vec![Dimension::Constant(784), Dimension::Constant(256)],
    DataType::FP32, MemoryLayout::Contiguous,
);
let b1_ty = ctx.make_tensor_type(
    vec![Dimension::Constant(256)],
    DataType::FP32, MemoryLayout::Contiguous,
);
let h1_ty = ctx.make_tensor_type(
    vec![Dimension::Constant(1), Dimension::Constant(256)],
    DataType::FP32, MemoryLayout::Contiguous,
);

let block = ctx.create_block();
let x  = ctx.create_block_arg(block, input_ty);
let w1 = ctx.create_block_arg(block, w1_ty);
let b1 = ctx.create_block_arg(block, b1_ty);

// MatMul
let (mm_op, mm_res) = ctx.create_op(
    "tensor.matmul", "tensor", vec![x, w1], vec![h1_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, mm_op);

// Add bias
let (add_op, add_res) = ctx.create_op(
    "tensor.add", "tensor", vec![mm_res[0], b1], vec![h1_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, add_op);

// ReLU
let (relu_op, _relu_res) = ctx.create_op(
    "tensor.relu", "tensor", vec![add_res[0]], vec![h1_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, relu_op);

lift_core::verifier::verify(&ctx).expect("Verification failed");
```

### 5.4 Tensor Fusion: Fuse MatMul + Bias + ReLU

**Problem:** Three separate GPU kernels waste memory bandwidth on intermediate results.

**Solution:** The `TensorFusion` pass detects `matmul → add → relu` and fuses them:

```rust
use lift_core::pass::PassManager;
use lift_opt::{Canonicalize, ConstantFolding, TensorFusion, DeadCodeElimination};

let mut pm = PassManager::new();
pm.add_pass(Box::new(Canonicalize));        // x + 0 → x, x * 1 → x
pm.add_pass(Box::new(ConstantFolding));     // fold constants at compile time
pm.add_pass(Box::new(TensorFusion));        // matmul + bias + relu → fused
pm.add_pass(Box::new(DeadCodeElimination)); // remove dead ops

let results = pm.run_all(&mut ctx);
for (name, result) in &results {
    println!("  {}: {:?}", name, result);
}
```

**Before fusion:**

```
%h1 = "tensor.matmul"(%x, %w1)  : (...) -> tensor<1x256xf32>
%h2 = "tensor.add"(%h1, %b1)    : (...) -> tensor<1x256xf32>
%h3 = "tensor.relu"(%h2)        : (...) -> tensor<1x256xf32>
```

**After fusion:**

```
%h3 = "tensor.fused_matmul_bias_relu"(%x, %w1, %b1) : (...) -> tensor<1x256xf32>
```

### 5.5 Analyse Resource Usage

```rust
use lift_sim::analysis::analyze_module;

let report = analyze_module(&ctx);
println!("Total ops: {}", report.num_ops);
println!("Tensor ops: {}", report.num_tensor_ops);
println!("Total FLOPs: {}", report.total_flops);
println!("Total memory: {} bytes", report.total_memory_bytes);
println!("Peak memory: {} bytes", report.peak_memory_bytes);

for (op_name, count) in &report.op_breakdown {
    println!("  {}: {}", op_name, count);
}
```

### 5.6 Shape Inference and FLOPs Counting

LIFT computes shapes and FLOPs for every tensor operation:

```rust
use lift_tensor::{TensorOp, ShapeInference};
use lift_core::types::{TensorTypeInfo, Dimension, DataType, MemoryLayout};

let a = TensorTypeInfo {
    shape: vec![Dimension::Constant(2), Dimension::Constant(3)],
    dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
};
let b = TensorTypeInfo {
    shape: vec![Dimension::Constant(3), Dimension::Constant(4)],
    dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
};

// Shape inference
let output = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).unwrap();
// output[0].shape = [2, 4]

// FLOPs: 2*M*N*K = 2*2*4*3 = 48
let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
assert_eq!(flops, Some(48));

// Memory bytes: input A + input B + output
let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
println!("Memory: {:?} bytes", mem);
```

### 5.7 All 107 Tensor Operations

| Category | Operations |
|----------|-----------|
| **Arithmetic** | add, sub, mul, div, neg, matmul, linear, conv2d, embedding |
| **Activations** | relu, gelu, silu, sigmoid, softmax, tanh, leaky_relu, elu, mish, hard_swish, hard_sigmoid |
| **Normalisation** | layernorm, rmsnorm, batchnorm, groupnorm, instancenorm |
| **Shape** | reshape, transpose, concat, split, gather, scatter, squeeze, unsqueeze, permute, expand, slice, pad, tile |
| **Constants** | constant, zeros, ones, arange, full |
| **Attention** | attention, multi_head_attention, multi_query_attention, grouped_query_attention, flash_attention, sliding_window_attention, cross_attention, paged_attention |
| **MoE** | moe_dispatch, moe_combine |
| **Convolution** | conv1d, conv3d, conv_transpose2d, depthwise_conv2d, dilated_conv2d |
| **Pooling** | maxpool2d, avgpool2d, adaptive_avgpool2d, global_avgpool |
| **Recurrent** | lstm_cell, gru_cell, rnn_cell |
| **Advanced Math** | einsum, fft, ifft, svd, eig, solve, topk, sort, cumsum, where, clamp |
| **Sparse** | sparse_matmul, sparse_embedding |
| **Quantisation** | quantize, dequantize, quantize_int4, dequantize_int4, quantize_fp8, dequantize_fp8 |
| **Diffusion** | unet_down_block, unet_up_block, timestep_embedding |
| **GNN** | gnn_message_passing, gnn_global_pooling |
| **Memory** | checkpoint, offload, grad_accumulate |
| **Gradient** | grad_matmul, grad_relu, grad_softmax, grad_layernorm, grad_attention, grad_conv2d, grad_linear, grad_gelu |
| **Parallelism** | parallel_split, parallel_allreduce, pipeline_send, pipeline_receive |
| **Fused** | fused_matmul_bias_relu, fused_matmul_bias, fused_linear_gelu, fused_attention_layernorm, fused_linear_silu, fused_conv_batchnorm_relu |

---

## 6. Use Case 2 — Transformer Attention and FlashAttention

### 6.1 Problem

Transformers use self-attention which scales O(n²) in memory. For long sequences (>512 tokens), this becomes the bottleneck.

### 6.2 Attention in `.lif`

File: `examples/attention.lif`

```
#dialect tensor

module @transformer {
    func @self_attention(%q: tensor<1x128x64xf32>, %k: tensor<1x128x64xf32>,
                         %v: tensor<1x128x64xf32>, %norm_w: tensor<64xf32>)
                         -> tensor<1x128x64xf32> {
        %attn = "tensor.attention"(%q, %k, %v)
            : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>)
            -> tensor<1x128x64xf32>
        %normed = "tensor.layernorm"(%attn, %norm_w)
            : (tensor<1x128x64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>
        return %normed
    }
}
```

### 6.3 FlashAttention Pass

The `FlashAttentionPass` replaces `tensor.attention` with `tensor.flash_attention` when sequence length exceeds a threshold:

```rust
use lift_opt::FlashAttentionPass;
use lift_core::pass::PassManager;

let mut pm = PassManager::new();
pm.add_pass(Box::new(FlashAttentionPass { seq_len_threshold: 512 }));
pm.run_all(&mut ctx);
// tensor.attention → tensor.flash_attention
// Same FLOPs, O(n) memory instead of O(n²)
```

### 6.4 Attention Variants

| Operation | Architecture | Memory |
|-----------|-------------|--------|
| `tensor.attention` | Standard QKV | O(n²) |
| `tensor.multi_head_attention` | GPT, BERT | O(n²) |
| `tensor.multi_query_attention` | PaLM | O(n²) reduced |
| `tensor.grouped_query_attention` | Llama 2 | O(n²) reduced |
| `tensor.flash_attention` | FlashAttention | **O(n)** |
| `tensor.sliding_window_attention` | Mistral | O(n×w) |
| `tensor.cross_attention` | Encoder-decoder | O(n×m) |
| `tensor.paged_attention` | vLLM KV cache | O(n) paged |

### 6.5 FLOPs Calculation

For attention: `FLOPs = 4 × B × H × S² × D`

```rust
use lift_tensor::{TensorOp, ShapeInference};
use lift_core::types::{TensorTypeInfo, Dimension, DataType, MemoryLayout};

let q = TensorTypeInfo {
    shape: vec![
        Dimension::Constant(1),    // batch
        Dimension::Constant(8),    // heads
        Dimension::Constant(2048), // seq_len
        Dimension::Constant(64),   // head_dim
    ],
    dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
};

let flops = ShapeInference::compute_flops(&TensorOp::Attention, &[&q, &q, &q]);
println!("Attention FLOPs: {:?}", flops);
// 4 × 1 × 8 × 2048 × 2048 × 64 ≈ 8.6 billion
```

---

## 7. Use Case 3 — Quantum Circuit Design and Noise Analysis

### 7.1 Problem

Design a quantum circuit, understand which gates your hardware supports, model noise, and estimate fidelity before executing on real devices.

### 7.2 Bell State in `.lif`

File: `examples/quantum_bell.lif`

```
#dialect quantum

module @bell_state {
    func @bell(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        return %q3, %q4
    }
}
```

### 7.3 Build a Circuit Programmatically

```rust
use lift_core::{Context, Attributes, Location};

let mut ctx = Context::new();
let qubit_ty = ctx.make_qubit_type();

let block = ctx.create_block();
let q0 = ctx.create_block_arg(block, qubit_ty);
let q1 = ctx.create_block_arg(block, qubit_ty);

// Hadamard on q0
let (h_op, h_res) = ctx.create_op(
    "quantum.h", "quantum",
    vec![q0], vec![qubit_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, h_op);

// CNOT on (q0', q1)
let (cx_op, cx_res) = ctx.create_op(
    "quantum.cx", "quantum",
    vec![h_res[0], q1], vec![qubit_ty, qubit_ty],
    Attributes::new(), Location::unknown(),
);
ctx.add_op_to_block(block, cx_op);

lift_core::verifier::verify(&ctx).expect("Valid circuit");
```

### 7.4 All 46+ Quantum Gates

| Category | Gates |
|----------|-------|
| **1Q standard** | H, X, Y, Z, S, Sdg, T, Tdg, SX |
| **1Q parametric** | RX, RY, RZ, P, U1, U2, U3 |
| **1Q fixed** | Rx90, Rx180 |
| **2Q standard** | CX, CZ, CY, SWAP, ISWAP, ECR |
| **2Q parametric** | RZX, XX, YY, ZZ, CPhase, XY, CP |
| **IonQ native** | GPI, GPI2, MS |
| **3Q** | CCX (Toffoli), CSWAP (Fredkin) |
| **Multi-controlled** | MCX, MCZ |
| **Measurement** | Measure, MeasureAll, Reset, Barrier, Init |
| **Special** | GlobalPhase, Delay, VirtualRZ, IfElse, ParamGate |

### 7.5 Hardware-Native Gate Sets

```rust
use lift_quantum::QuantumGate;
use lift_quantum::gates::Provider;

// IBM Eagle/Kyoto native: {RZ, SX, X, CX, ECR}
let ibm = QuantumGate::native_basis(Provider::IbmEagle);

// Rigetti native: {RZ, RX, CZ, CPhase, XY}
let rigetti = QuantumGate::native_basis(Provider::Rigetti);

// IonQ native: {GPI, GPI2, MS}
let ionq = QuantumGate::native_basis(Provider::IonQ);

// Quantinuum native: {RZ, RX, RY, ZZ}
let quantinuum = QuantumGate::native_basis(Provider::Quantinuum);

for gate in ibm {
    println!("{} ({} qubit, parametric: {}, clifford: {})",
        gate.op_name(), gate.num_qubits(),
        gate.is_parametric(), gate.is_clifford());
}
```

### 7.6 Noise Models

**Problem:** Real quantum hardware introduces errors. You need to model them before execution.

```rust
use lift_quantum::{NoiseModel, GateNoise, CircuitNoise};

// Depolarizing noise (1Q gate error p = 0.001)
let noise_1q = NoiseModel::Depolarizing { p: 0.001 };
println!("1Q fidelity: {:.6}", noise_1q.fidelity()); // 0.999000

// Thermal relaxation
let thermal = NoiseModel::ThermalRelaxation {
    t1_us: 100.0, t2_us: 80.0, gate_time_us: 0.3,
};
println!("Thermal fidelity: {:.6}", thermal.fidelity());

// Composed noise
let combined = noise_1q.compose(&thermal);
println!("Combined fidelity: {:.6}", combined.fidelity());

// Track noise across a full circuit
let mut circuit = CircuitNoise::new();
let g1q = GateNoise::with_depolarizing(0.999, 0.02);
let g2q = GateNoise::with_depolarizing(0.99, 0.3);

circuit.add_gate(&g1q, false);  // H  (1Q)
circuit.add_gate(&g2q, true);   // CX (2Q)

println!("Circuit fidelity: {:.6}", circuit.total_fidelity);
println!("Gate count: {}, 2Q gates: {}", circuit.gate_count, circuit.two_qubit_count);
println!("Meets 99% threshold: {}", circuit.meets_threshold(0.99));
```

All noise models:

| Model | Parameter | Use Case |
|-------|-----------|----------|
| `Ideal` | — | Simulation baseline |
| `Depolarizing { p }` | Error probability | General gate errors |
| `AmplitudeDamping { gamma }` | Decay rate | T1 relaxation |
| `PhaseDamping { gamma }` | Dephasing rate | T2 dephasing |
| `BitFlip { p }` | Flip probability | Classical-like errors |
| `PhaseFlip { p }` | Phase flip prob | Z errors |
| `ThermalRelaxation { t1, t2, t }` | Coherence times | Realistic hardware |
| `Kraus { operators }` | Kraus matrices | Custom channels |
| `Composed(vec)` | Multiple models | Layered noise |

### 7.7 Quantum Cost Model

```rust
use lift_sim::cost::QuantumCostModel;

// Superconducting (IBM-like): fast but lower fidelity
let sc = QuantumCostModel::superconducting_default();
// gate_time_1q: 0.02μs, gate_time_2q: 0.3μs, fidelity_1q: 0.999, fidelity_2q: 0.99

// Trapped-ion (IonQ-like): slow but very high fidelity
let ti = QuantumCostModel::trapped_ion_default();
// gate_time_1q: 10μs, gate_time_2q: 200μs, fidelity_1q: 0.9999, fidelity_2q: 0.999

// Neutral-atom: fast with moderate fidelity, many qubits
let na = QuantumCostModel::neutral_atom_default();
// gate_time_1q: 0.5μs, gate_time_2q: 1.0μs, fidelity_1q: 0.999, fidelity_2q: 0.995

// Compare fidelity for a 100-gate circuit (80×1Q + 20×2Q)
println!("Superconducting: {:.6}", sc.circuit_fidelity(80, 20));
println!("Trapped-ion:     {:.6}", ti.circuit_fidelity(80, 20));
println!("Neutral-atom:    {:.6}", na.circuit_fidelity(80, 20));
```

### 7.8 Gate Optimisation Passes

```rust
use lift_opt::{GateCancellation, RotationMerge, NoiseAwareSchedule, LayoutMapping};
use lift_core::pass::PassManager;

let mut pm = PassManager::new();
pm.add_pass(Box::new(GateCancellation));     // H·H → I, X·X → I
pm.add_pass(Box::new(RotationMerge));        // Rz(a)·Rz(b) → Rz(a+b)
pm.add_pass(Box::new(NoiseAwareSchedule));   // schedule to minimise noise
pm.add_pass(Box::new(LayoutMapping));        // map to device topology

let results = pm.run_all(&mut ctx);
for (name, result) in &results {
    println!("  {}: {:?}", name, result);
}
```

| Pass | What It Does |
|------|-------------|
| `GateCancellation` | Cancels adjacent inverse gates (H·H, X·X, etc.) |
| `RotationMerge` | Merges consecutive rotations: Rz(a)·Rz(b) → Rz(a+b) |
| `NoiseAwareSchedule` | Reorders gates to place noisy 2Q gates on high-fidelity edges |
| `LayoutMapping` | Maps logical qubits to physical qubits with SWAP insertion |

---

## 8. Use Case 4 — Hybrid Classical-Quantum (VQE)

### 8.1 Problem

VQE (Variational Quantum Eigensolver) is a hybrid algorithm requiring:
1. Encoding classical data into quantum states
2. Running a parametrised circuit (ansatz)
3. Computing gradients of quantum parameters
4. Iterating with a classical optimiser

### 8.2 Encoding Strategies

```rust
use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

// Angle encoding: 1 qubit per feature, circuit depth 1
let angle = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);
println!("Qubits: {}, depth: {}", angle.num_qubits,
    angle.strategy.circuit_depth(4));
// 4 qubits, depth 1

// Amplitude encoding: log2(N) qubits, depth N
let amp = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 16);
println!("Qubits: {}, depth: {}", amp.num_qubits,
    amp.strategy.circuit_depth(16));
// 4 qubits, depth 16

// IQP encoding: N qubits, depth 2N
let iqp = EncodingConfig::new(EncodingStrategy::IQPEncoding, 8);
println!("Qubits: {}, depth: {}", iqp.num_qubits,
    iqp.strategy.circuit_depth(8));
// 8 qubits, depth 16
```

| Strategy | Qubits | Depth | Best For |
|----------|--------|-------|----------|
| `AngleEncoding` | N | 1 | Small feature spaces |
| `AmplitudeEncoding` | log₂(N) | N | Large feature spaces |
| `BasisEncoding` | N | 1 | Binary data |
| `IQPEncoding` | N | 2N | Quantum advantage proofs |
| `HamiltonianEncoding` | N | N | Physics simulations |
| `KernelEncoding` | N | 3N | Quantum kernel methods |

### 8.3 Gradient Methods

```rust
use lift_hybrid::gradient::GradientMethod;

let num_params = 20;

// Parameter shift: exact, 2 evaluations per parameter
let ps = GradientMethod::ParameterShift;
println!("Evals: {}, exact: {}", ps.circuit_evaluations(num_params), ps.is_exact());
// 40, true

// SPSA: stochastic, only 2 evaluations total
let spsa = GradientMethod::SPSA;
println!("Evals: {}, exact: {}", spsa.circuit_evaluations(num_params), spsa.is_exact());
// 2, false

// Adjoint: exact, 1 evaluation (best for simulators)
let adj = GradientMethod::Adjoint;
println!("Evals: {}, exact: {}", adj.circuit_evaluations(num_params), adj.is_exact());
// 1, true
```

| Method | Evaluations | Exact | Best For |
|--------|------------|-------|----------|
| `ParameterShift` | 2N | Yes | Hardware |
| `FiniteDifference` | N+1 | No | Quick approximation |
| `SPSA` | 2 | No | Many parameters |
| `Adjoint` | 1 | Yes | Simulators |
| `Backprop` | 1 | Yes | Classical parts |

### 8.4 Joint Gradient (Classical + Quantum)

```rust
use lift_hybrid::gradient::{GradientMethod, JointGradientConfig};

let config = JointGradientConfig {
    classical_method: GradientMethod::Backprop,
    quantum_method: GradientMethod::ParameterShift,
    num_classical_params: 1000,
    num_quantum_params: 20,
};

println!("Total evaluations: {}", config.total_evaluations());
// 1 (backprop) + 40 (param shift) = 41
```

### 8.5 VQE Pipeline in `.lif`

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @vqe {
    func @step(%data: tensor<1x4xf32>, %q0: qubit, %q1: qubit) -> f32 {
        // 1. Encode classical data
        %encoded = "hybrid.encode"(%data) : (tensor<1x4xf32>) -> tensor<1x4xf32>

        // 2. Variational ansatz
        %q2 = "quantum.ry"(%q0) {angle = 0.5} : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        %q5 = "quantum.rz"(%q3) {angle = 1.2} : (qubit) -> qubit

        // 3. Measure expectation value
        %energy = "hybrid.measure_expectation"(%q5) : (qubit) -> f32

        // 4. Compute gradient
        %grad = "hybrid.parameter_shift"(%energy) : (f32) -> f32

        return %energy
    }
}
```

### 8.6 All 21 Hybrid Operations

| Operation | Description |
|-----------|-------------|
| `hybrid.encode` | Encode classical data into quantum state |
| `hybrid.decode` | Decode quantum measurement to classical |
| `hybrid.parameter_shift` | Gradient via parameter shift rule |
| `hybrid.finite_difference` | Gradient via finite differences |
| `hybrid.spsa` | Stochastic parameter shift approximation |
| `hybrid.adjoint_diff` | Gradient via adjoint differentiation |
| `hybrid.stochastic_param_shift` | Stochastic parameter shift |
| `hybrid.joint_gradient` | Joint classical+quantum gradient |
| `hybrid.classical_preprocess` | Classical preprocessing step |
| `hybrid.quantum_postprocess` | Quantum postprocessing step |
| `hybrid.forward` | Hybrid forward pass |
| `hybrid.backward` | Hybrid backward pass |
| `hybrid.vqc_layer` | Variational quantum circuit layer |
| `hybrid.vqe_ansatz` | VQE ansatz circuit |
| `hybrid.qaoa_layer` | QAOA mixer + cost layer |
| `hybrid.quantum_kernel` | Quantum kernel evaluation |
| `hybrid.gpu_to_qpu` | Transfer data GPU → QPU |
| `hybrid.qpu_to_gpu` | Transfer data QPU → GPU |
| `hybrid.co_execute` | Co-execute classical and quantum |
| `hybrid.measure_expectation` | Measure observable expectation |
| `hybrid.measure_samples` | Measure and return bit-strings |

---

## 9. Use Case 5 — Model Import

### 9.1 Problem

You have existing models in ONNX, PyTorch FX, or OpenQASM format and want to bring them into LIFT for unified optimisation and analysis.

### 9.2 ONNX Import

```rust
use lift_import::OnnxImporter;
use lift_core::pass::PassManager;

let importer = OnnxImporter::new();
let mut ctx = importer.import("model.onnx")
    .expect("ONNX import failed");

// Optimise with LIFT
let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::TensorFusion));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.run_all(&mut ctx);
```

Supported ONNX operators are mapped to LIFT tensor ops (MatMul, Conv, ReLU, Softmax, Attention, etc.).

### 9.3 PyTorch FX Import

```rust
use lift_import::PyTorchFxImporter;

let importer = PyTorchFxImporter::new();
let ctx = importer.import("model_fx.json")
    .expect("FX import failed");
```

Import from `torch.fx` graph JSON exports. All standard PyTorch operations are mapped to their LIFT equivalents.

### 9.4 OpenQASM 3.0 Import

```rust
use lift_import::OpenQasm3Importer;
use lift_core::pass::PassManager;

let importer = OpenQasm3Importer::new();
let mut ctx = importer.import("circuit.qasm")
    .expect("QASM import failed");

// Optimise the quantum circuit
let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::GateCancellation));
pm.add_pass(Box::new(lift_opt::RotationMerge));
pm.run_all(&mut ctx);
```

### 9.5 Import → Analyse → Compare

A common workflow: import a model, analyse it, optimise, then compare before/after:

```rust
use lift_sim::analysis::analyze_module;

// Before optimisation
let report_before = analyze_module(&ctx);
println!("Before: {} ops, {} FLOPs", report_before.num_ops, report_before.total_flops);

// Run passes...
pm.run_all(&mut ctx);

// After optimisation
let report_after = analyze_module(&ctx);
println!("After:  {} ops, {} FLOPs", report_after.num_ops, report_after.total_flops);
println!("Ops reduced: {:.1}%",
    (1.0 - report_after.num_ops as f64 / report_before.num_ops as f64) * 100.0);
```

---

## 10. Use Case 6 — Performance Prediction

### 10.1 Problem

Before running a model on expensive hardware, you need to know:
- How long will it take?
- Is it compute-bound or memory-bound?
- Will it fit in GPU memory?
- How many GPUs are needed?

### 10.2 Roofline Model (Classical)

```rust
use lift_sim::analysis::analyze_module;
use lift_sim::cost::CostModel;
use lift_predict::roofline::predict_performance;

let report = analyze_module(&ctx);

// NVIDIA A100
let a100 = CostModel::a100();
let pred_a100 = predict_performance(&report, &a100);

println!("=== A100 Prediction ===");
println!("Compute time: {:.4} ms", pred_a100.compute_time_ms);
println!("Memory time:  {:.4} ms", pred_a100.memory_time_ms);
println!("Predicted:    {:.4} ms", pred_a100.predicted_time_ms);
println!("Arithmetic intensity: {:.2} FLOP/byte", pred_a100.arithmetic_intensity);
println!("Bottleneck: {}", pred_a100.bottleneck); // "compute" or "memory"

// NVIDIA H100
let h100 = CostModel::h100();
let pred_h100 = predict_performance(&report, &h100);

println!("\n=== H100 Prediction ===");
println!("Predicted: {:.4} ms", pred_h100.predicted_time_ms);
println!("Speedup vs A100: {:.2}x",
    pred_a100.predicted_time_ms / pred_h100.predicted_time_ms);
```

### 10.3 GPU Profiles

| Profile | TFLOPS (FP16) | Memory BW (GB/s) | VRAM | TDP |
|---------|--------------|-------------------|------|-----|
| `CostModel::a100()` | 312 | 2,039 | 80 GB | 400W |
| `CostModel::h100()` | 989 | 3,350 | 80 GB | 700W |

### 10.4 Memory Fit and Multi-GPU Planning

```rust
let model = CostModel::a100();
let bytes = report.total_memory_bytes;

println!("Model size: {:.2} GB", bytes as f64 / 1e9);
println!("Fits in 1 GPU: {}", model.fits_in_memory(bytes));
println!("GPUs needed: {}", model.num_gpus_needed(bytes));

// Arithmetic intensity analysis
let ai = model.arithmetic_intensity(report.total_flops, bytes);
let ridge = model.flops_per_second / (model.memory_bandwidth_gb_s * 1e9);
println!("Arithmetic intensity: {:.2} FLOP/byte", ai);
println!("Ridge point: {:.2} FLOP/byte", ridge);
println!("Regime: {}", if ai >= ridge { "compute-bound" } else { "memory-bound" });
```

### 10.5 Quantum Performance Prediction

```rust
use lift_predict::roofline::predict_quantum;
use lift_sim::quantum_sim::QuantumAnalysis;
use lift_sim::cost::QuantumCostModel;

let analysis = QuantumAnalysis {
    num_qubits_used: 10,
    gate_count: 200,
    one_qubit_gates: 150,
    two_qubit_gates: 50,
    measurements: 10,
    circuit_depth: 30,
    estimated_fidelity: 0.92,
};

let sc = QuantumCostModel::superconducting_default();
let prediction = predict_quantum(&analysis, &sc, 0.01); // 1% precision

println!("Estimated fidelity: {:.6}", prediction.estimated_fidelity);
println!("Circuit time: {:.2} μs", prediction.circuit_time_us);
println!("Shots for 1%% precision: {}", prediction.num_shots_for_precision);
println!("Total execution: {:.2} ms", prediction.total_execution_time_ms);

// Compare technologies
let ti = QuantumCostModel::trapped_ion_default();
let pred_ti = predict_quantum(&analysis, &ti, 0.01);
println!("\nTrapped-ion fidelity: {:.6} (vs {:.6} superconducting)",
    pred_ti.estimated_fidelity, prediction.estimated_fidelity);
println!("Trapped-ion time: {:.2} ms (vs {:.2} ms)",
    pred_ti.total_execution_time_ms, prediction.total_execution_time_ms);
```

---

## 11. Use Case 7 — Quantised Inference

### 11.1 Problem

FP32 models are too large and slow for deployment. You want INT8, INT4, or FP8 for faster inference.

### 11.2 Quantisation Operations

| Operation | Conversion |
|-----------|-----------|
| `tensor.quantize` | FP32 → INT8 |
| `tensor.dequantize` | INT8 → FP32 |
| `tensor.quantize_int4` | FP32 → INT4 |
| `tensor.dequantize_int4` | INT4 → FP32 |
| `tensor.quantize_fp8` | FP32 → FP8 |
| `tensor.dequantize_fp8` | FP8 → FP32 |

### 11.3 Quantised Inference in `.lif`

```
#dialect tensor

module @quantised_inference {
    func @forward(%x: tensor<1x784xf32>,
                  %w1_q: tensor<784x256xi8>,
                  %b1: tensor<256xf32>) -> tensor<1x256xf32> {
        // Dequantize INT8 weights to FP32
        %w1 = "tensor.dequantize"(%w1_q) : (tensor<784x256xi8>) -> tensor<784x256xf32>

        // Compute in FP32
        %h1 = "tensor.matmul"(%x, %w1) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
        %h2 = "tensor.add"(%h1, %b1) : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
        %out = "tensor.relu"(%h2) : (tensor<1x256xf32>) -> tensor<1x256xf32>
        return %out
    }
}
```

### 11.4 Automatic Quantisation Pass

The `QuantisationPass` annotates ops that are safe to quantise:

```rust
use lift_opt::QuantisationPass;
use lift_core::pass::PassManager;

let mut pm = PassManager::new();
pm.add_pass(Box::new(QuantisationPass));
pm.run_all(&mut ctx);
```

### 11.5 Memory Savings

| Data Type | Bits | Size vs FP32 | Use Case |
|-----------|------|-------------|----------|
| FP32 | 32 | 1× baseline | Training |
| FP16 / BF16 | 16 | 0.5× | Mixed-precision training |
| FP8 (E4M3) | 8 | 0.25× | H100 inference |
| INT8 | 8 | 0.25× | Server inference |
| INT4 | 4 | 0.125× | Edge/mobile inference |
| INT2 | 2 | 0.0625× | Extreme compression |

### 11.6 FP8 Formats

LIFT supports both FP8 variants:

```rust
use lift_tensor::ops::Fp8Format;

// E4M3: 4 exponent, 3 mantissa — higher precision, smaller range
// Best for: weights and activations in forward pass
let e4m3 = Fp8Format::E4M3;

// E5M2: 5 exponent, 2 mantissa — lower precision, larger range
// Best for: gradients in backward pass
let e5m2 = Fp8Format::E5M2;
```

---

## 12. Use Case 8 — Backend Export

### 12.1 Problem

After optimisation, you need to compile the IR to executable code for GPU/CPU or quantum hardware.

### 12.2 Export to LLVM IR

```rust
use lift_export::LlvmExporter;

let exporter = LlvmExporter::new();
let llvm_ir = exporter.export(&ctx).expect("LLVM export failed");

std::fs::write("output.ll", &llvm_ir).unwrap();
println!("Written {} bytes of LLVM IR", llvm_ir.len());
```

Compile the output:

```bash
# Compile to binary
clang -O3 output.ll -o model

# Or to object file
llc -O3 output.ll -filetype=obj -o model.o
```

### 12.3 Export to OpenQASM 3.0

```rust
use lift_export::QasmExporter;

let exporter = QasmExporter::new();
let qasm = exporter.export(&ctx).expect("QASM export failed");

std::fs::write("circuit.qasm", &qasm).unwrap();
```

The output is standard OpenQASM 3.0 executable on:
- **IBM Quantum** (via Qiskit)
- **Rigetti** (via pyQuil)
- **IonQ** (via native API)
- **Quantinuum** (via TKET)
- Any OpenQASM 3.0 compatible platform

### 12.4 Full Export Pipeline

```rust
use lift_core::printer::print_ir;

// Print human-readable IR (for debugging)
let ir_text = print_ir(&ctx);
std::fs::write("debug.lif", &ir_text).unwrap();

// Export to LLVM (for tensor/classical ops)
let llvm = LlvmExporter::new().export(&ctx).expect("LLVM failed");
std::fs::write("model.ll", &llvm).unwrap();

// Export to QASM (for quantum ops)
let qasm = QasmExporter::new().export(&ctx).expect("QASM failed");
std::fs::write("circuit.qasm", &qasm).unwrap();
```

---

## 13. Use Case 9 — Energy and Carbon Estimation

### 13.1 Problem

AI training and inference consume significant energy. You want to estimate the environmental impact before committing resources.

### 13.2 Classical Energy Model

```rust
use lift_sim::cost::{CostModel, EnergyModel};
use lift_sim::analysis::analyze_module;
use lift_predict::roofline::predict_performance;

let report = analyze_module(&ctx);
let cost = CostModel::h100();
let prediction = predict_performance(&report, &cost);

let energy = EnergyModel::h100();

// Single inference
let joules = energy.energy_joules(prediction.predicted_time_ms, 1);
let kwh = energy.energy_kwh(prediction.predicted_time_ms, 1);
let co2 = energy.carbon_grams(prediction.predicted_time_ms, 1);

println!("Single inference:");
println!("  Energy: {:.4} J ({:.8} kWh)", joules, kwh);
println!("  CO₂: {:.6} g", co2);

// Training: 8 GPUs for 72 hours
let train_ms = 72.0 * 3600.0 * 1000.0;
let train_kwh = energy.energy_kwh(train_ms, 8);
let train_co2_kg = energy.carbon_grams(train_ms, 8) / 1000.0;

println!("\nTraining (8× H100, 72h):");
println!("  Energy: {:.2} kWh", train_kwh);
println!("  CO₂: {:.2} kg", train_co2_kg);
println!("  Equivalent to: {:.0} km driven", train_co2_kg / 0.21);
```

### 13.3 Energy Profiles

| Profile | GPU TDP | CPU TDP | Cooling PUE | CO₂ (g/kWh) |
|---------|---------|---------|-------------|--------------|
| `EnergyModel::a100()` | 400W | 250W | 1.1 | 400 (world avg) |
| `EnergyModel::h100()` | 700W | 350W | 1.1 | 400 (world avg) |

### 13.4 Quantum Energy Estimation

```rust
let energy = EnergyModel::h100();

// Quantum circuit: dominated by cryogenic cooling
let circuit_time_us = 100.0;
let num_qubits = 127;
let quantum_joules = energy.quantum_energy_joules(circuit_time_us, num_qubits);

println!("Quantum energy: {:.4} J", quantum_joules);
println!("  Cryogenics: ~25 kW (dilution refrigerator)");
println!("  Control electronics: ~{:.0} W ({} qubits × 10W)", num_qubits as f64 * 10.0, num_qubits);
```

### 13.5 Compare Classical vs Quantum Energy

```rust
// Classical: matmul 1000×1000 on H100
let classical_time_ms = cost.compute_time_ms(2 * 1000 * 1000 * 1000);
let classical_j = energy.energy_joules(classical_time_ms, 1);

// Quantum: 100-gate circuit
let quantum_j = energy.quantum_energy_joules(100.0, 50);

println!("Classical (1000×1000 matmul): {:.4} J", classical_j);
println!("Quantum (100-gate circuit):   {:.4} J", quantum_j);
println!("Note: Quantum energy is dominated by cryogenic overhead,");
println!("not by the computation itself.");
```

---

## 14. Use Case 10 — Device Topology and Routing

### 14.1 Problem

Quantum hardware has limited connectivity — not all qubits can directly interact. Two-qubit gates between non-adjacent qubits require SWAP operations, increasing circuit depth and noise.

### 14.2 Built-in Topologies

```rust
use lift_quantum::DeviceTopology;

// Linear chain (nearest-neighbour)
let linear = DeviceTopology::linear(10);
println!("Linear: {} qubits, {} edges, diameter {}",
    linear.num_qubits, linear.edges.len(), linear.diameter());

// 2D Grid (superconducting chips)
let grid = DeviceTopology::grid(4, 4);
println!("Grid 4×4: {} qubits, avg connectivity {:.2}",
    grid.num_qubits, grid.avg_connectivity());

// IBM Heavy-hex (Eagle/Heron processors)
let heavy_hex = DeviceTopology::heavy_hex(127);
println!("Heavy-hex: {} qubits, {} edges",
    heavy_hex.num_qubits, heavy_hex.edges.len());

// All-to-all (trapped-ion systems)
let ion = DeviceTopology::all_to_all(32);
println!("All-to-all: {} qubits, {} edges, diameter {}",
    ion.num_qubits, ion.edges.len(), ion.diameter());

// Binary tree
let tree = DeviceTopology::tree(15);

// Custom topology
let custom = DeviceTopology::custom("my_chip",
    &[(0,1), (1,2), (2,3), (0,3), (1,3)], 0.995);
```

| Topology | Constructor | Typical Hardware |
|----------|-----------|-----------------|
| Linear | `linear(n)` | Simple chains |
| Grid | `grid(rows, cols)` | Google Sycamore |
| Heavy-hex | `heavy_hex(n)` | IBM Eagle/Heron |
| All-to-all | `all_to_all(n)` | IonQ, Quantinuum |
| Tree | `tree(n)` | Hierarchical architectures |
| Custom | `custom(name, edges, fidelity)` | Any device |

### 14.3 Routing and SWAP Cost

```rust
let topo = DeviceTopology::grid(5, 5);

// Are two qubits directly connected?
println!("0↔1 connected: {}", topo.are_connected(0, 1));  // true
println!("0↔6 connected: {}", topo.are_connected(0, 6));  // false

// Find shortest path between qubits
if let Some(path) = topo.shortest_path(0, 24) {
    println!("Path 0→24: {:?}", path);
    println!("SWAPs needed: {}", path.len() - 2);
}

// Number of SWAPs between any two qubits
let swaps = topo.swap_distance(0, 24);
println!("SWAP distance 0→24: {:?}", swaps);

// Neighbours of a qubit
println!("Neighbours of qubit 12: {:?}", topo.neighbors(12));

// Graph metrics
println!("Diameter: {}", topo.diameter());
println!("Avg connectivity: {:.2}", topo.avg_connectivity());
```

### 14.4 Layout Mapping Pass

The `LayoutMapping` pass automatically inserts SWAPs to match your device:

```rust
use lift_opt::LayoutMapping;
use lift_core::pass::PassManager;

let mut pm = PassManager::new();
pm.add_pass(Box::new(LayoutMapping));
pm.run_all(&mut ctx);
```

---

## 15. Use Case 11 — Diffusion and GNN Models

### 15.1 Diffusion Models (Stable Diffusion)

**Problem:** Diffusion models use UNet architectures with timestep conditioning and cross-attention. Standard tensor frameworks lack first-class support.

```
#dialect tensor

module @unet_step {
    func @denoise(%x: tensor<1x4x64x64xf32>, %t: tensor<1xf32>,
                  %context: tensor<1x77x768xf32>) -> tensor<1x4x64x64xf32> {
        // Timestep embedding
        %t_emb = "tensor.timestep_embedding"(%t)
            : (tensor<1xf32>) -> tensor<1x320xf32>

        // UNet down block (conv + attention)
        %d1 = "tensor.unet_down_block"(%x, %t_emb)
            : (tensor<1x4x64x64xf32>, tensor<1x320xf32>) -> tensor<1x320x32x32xf32>

        // Cross-attention with text context
        %attn = "tensor.cross_attention"(%d1, %context, %context)
            : (tensor<1x320x32x32xf32>, tensor<1x77x768xf32>, tensor<1x77x768xf32>)
            -> tensor<1x320x32x32xf32>

        // UNet up block (transpose conv + skip connections)
        %u1 = "tensor.unet_up_block"(%attn, %t_emb)
            : (tensor<1x320x32x32xf32>, tensor<1x320xf32>) -> tensor<1x4x64x64xf32>

        return %u1
    }
}
```

Diffusion-specific ops:

| Operation | Description |
|-----------|-------------|
| `tensor.timestep_embedding` | Sinusoidal timestep encoding |
| `tensor.unet_down_block` | Downsample with residual + attention |
| `tensor.unet_up_block` | Upsample with skip connections |

### 15.2 Graph Neural Networks (GNN)

**Problem:** GNNs operate on irregular graph structures. Message passing between nodes requires specialised aggregation operations.

```
#dialect tensor

module @gcn {
    func @forward(%nodes: tensor<100x64xf32>,
                  %edges: tensor<2x500xi64>,
                  %w: tensor<64x32xf32>) -> tensor<100x32xf32> {
        // Message passing: aggregate neighbour features
        %msg = "tensor.gnn_message_passing"(%nodes, %edges)
            {aggregation = "mean"}
            : (tensor<100x64xf32>, tensor<2x500xi64>) -> tensor<100x64xf32>

        // Linear transform
        %h = "tensor.matmul"(%msg, %w)
            : (tensor<100x64xf32>, tensor<64x32xf32>) -> tensor<100x32xf32>
        %out = "tensor.relu"(%h)
            : (tensor<100x32xf32>) -> tensor<100x32xf32>
        return %out
    }

    func @graph_classify(%nodes: tensor<100x32xf32>) -> tensor<1x32xf32> {
        // Global pooling: graph-level representation
        %graph = "tensor.gnn_global_pooling"(%nodes)
            {aggregation = "mean"}
            : (tensor<100x32xf32>) -> tensor<1x32xf32>
        return %graph
    }
}
```

GNN operations:

| Operation | Description | Aggregation |
|-----------|-------------|-------------|
| `tensor.gnn_message_passing` | Neighbour feature aggregation | sum, mean, max, min |
| `tensor.gnn_global_pooling` | Graph-level readout | sum, mean, max |

---

## 16. Use Case 12 — Budget-Constrained Compilation

### 16.1 Problem

You have hard resource constraints: maximum FLOPs, memory, time, or minimum quantum fidelity. You want to enforce these during compilation.

### 16.2 Static Budget

```rust
use lift_sim::cost::Budget;
use lift_sim::analysis::analyze_module;

let budget = Budget {
    max_flops: Some(1_000_000_000),          // 1 GFLOP
    max_memory_bytes: Some(1_073_741_824),   // 1 GB
    max_time_ms: Some(100.0),                // 100 ms
    min_fidelity: Some(0.99),                // 99% fidelity
    max_circuit_depth: Some(100),
};

let report = analyze_module(&ctx);

match budget.check_flops(report.total_flops) {
    Ok(()) => println!("FLOP budget OK"),
    Err(e) => println!("WARNING: {}", e),
}

match budget.check_memory(report.total_memory_bytes) {
    Ok(()) => println!("Memory budget OK"),
    Err(e) => println!("WARNING: {}", e),
}
```

### 16.3 Reactive Budget (Dynamic Tracking)

For iterative algorithms (VQE, QAOA) where resources are consumed over time:

```rust
use lift_sim::cost::{Budget, ReactiveBudget};

let budget = Budget {
    max_flops: Some(10_000_000_000),       // 10 GFLOP
    max_memory_bytes: Some(4_294_967_296), // 4 GB
    max_time_ms: Some(5000.0),             // 5 seconds
    min_fidelity: Some(0.90),              // 90% fidelity
    max_circuit_depth: None,
};

let mut tracker = ReactiveBudget::new(budget);

for iteration in 0..100 {
    // Simulate consuming resources each iteration
    tracker.consume(
        100_000_000,  // 100M FLOPs
        500_000_000,  // 500MB memory
        50.0,         // 50ms
        0.999,        // fidelity factor
    );

    match tracker.check_remaining() {
        Ok(()) => {
            let util = tracker.utilisation();
            if iteration % 10 == 0 {
                println!("Iter {}: FLOP {:.0}%, time {:.0}%",
                    iteration,
                    util.flop_ratio.unwrap_or(0.0) * 100.0,
                    util.time_ratio.unwrap_or(0.0) * 100.0,
                );
            }
        }
        Err(e) => {
            println!("Budget exceeded at iteration {}: {}", iteration, e);
            break;
        }
    }
}

// Query remaining budget
if let Some(remaining) = tracker.remaining_flops() {
    println!("Remaining FLOPs: {}", remaining);
}
if let Some(remaining) = tracker.remaining_time_ms() {
    println!("Remaining time: {:.2} ms", remaining);
}
```

---

## 17. Use Case 13 — End-to-End Pipelines

### 17.1 AI Pipeline: Import → Verify → Optimise → Predict → Export

```rust
use lift_import::OnnxImporter;
use lift_core::{verifier, pass::PassManager};
use lift_sim::analysis::analyze_module;
use lift_sim::cost::CostModel;
use lift_predict::roofline::predict_performance;
use lift_export::LlvmExporter;

// ── 1. Import ──
let mut ctx = OnnxImporter::new()
    .import("model.onnx").expect("Import failed");

// ── 2. Verify ──
verifier::verify(&ctx).expect("Verification failed");

// ── 3. Analyse (before) ──
let before = analyze_module(&ctx);

// ── 4. Optimise ──
let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::ConstantFolding));
pm.add_pass(Box::new(lift_opt::TensorFusion));
pm.add_pass(Box::new(lift_opt::FlashAttentionPass::default()));
pm.add_pass(Box::new(lift_opt::CommonSubexprElimination));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

for (name, result) in pm.run_all(&mut ctx) {
    println!("  {}: {:?}", name, result);
}

// ── 5. Analyse (after) ──
let after = analyze_module(&ctx);
println!("Ops: {} → {} ({:.1}% reduction)",
    before.num_ops, after.num_ops,
    (1.0 - after.num_ops as f64 / before.num_ops as f64) * 100.0);

// ── 6. Predict ──
let h100 = CostModel::h100();
let prediction = predict_performance(&after, &h100);
println!("H100: {:.4} ms ({}-bound)", prediction.predicted_time_ms, prediction.bottleneck);

// ── 7. Export ──
let llvm = LlvmExporter::new().export(&ctx).expect("Export failed");
std::fs::write("model.ll", &llvm).unwrap();
println!("Exported {} bytes of LLVM IR", llvm.len());
```

### 17.2 Quantum Pipeline: Parse → Verify → Optimise → Predict → Export

```rust
use lift_ast::{Lexer, Parser, IrBuilder};
use lift_core::{Context, verifier, pass::PassManager};
use lift_sim::cost::QuantumCostModel;
use lift_export::QasmExporter;

// ── 1. Parse ──
let source = std::fs::read_to_string("circuit.lif").unwrap();
let tokens = Lexer::new(&source).tokenize().to_vec();
let program = Parser::new(tokens).parse().unwrap();
let mut ctx = Context::new();
IrBuilder::new().build_program(&mut ctx, &program).unwrap();

// ── 2. Verify (SSA + linearity) ──
verifier::verify(&ctx).expect("Circuit verification failed");

// ── 3. Optimise ──
let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::GateCancellation));
pm.add_pass(Box::new(lift_opt::RotationMerge));
pm.add_pass(Box::new(lift_opt::NoiseAwareSchedule));
pm.add_pass(Box::new(lift_opt::LayoutMapping));

for (name, result) in pm.run_all(&mut ctx) {
    println!("  {}: {:?}", name, result);
}

// ── 4. Predict fidelity ──
let sc = QuantumCostModel::superconducting_default();
let fidelity = sc.circuit_fidelity(30, 10); // 30×1Q + 10×2Q
println!("Predicted fidelity: {:.6}", fidelity);

// ── 5. Export to OpenQASM 3.0 ──
let qasm = QasmExporter::new().export(&ctx).unwrap();
std::fs::write("circuit.qasm", &qasm).unwrap();
```

### 17.3 Hybrid Pipeline: VQE with Energy Estimation

```rust
use lift_core::{Context, Attributes, Location};
use lift_sim::cost::{CostModel, QuantumCostModel, EnergyModel, Budget, ReactiveBudget};
use lift_hybrid::gradient::GradientMethod;
use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

// ── Setup ──
let encoding = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);
let gradient = GradientMethod::ParameterShift;
let num_params = 12;

println!("Encoding: {} qubits, depth {}",
    encoding.num_qubits, encoding.strategy.circuit_depth(4));
println!("Gradient: {} evaluations per iteration",
    gradient.circuit_evaluations(num_params));

// ── Budget ──
let budget = Budget {
    max_flops: None,
    max_memory_bytes: None,
    max_time_ms: Some(60_000.0),  // 60 seconds
    min_fidelity: Some(0.80),
    max_circuit_depth: None,
};
let mut tracker = ReactiveBudget::new(budget);

// ── Cost models ──
let qcm = QuantumCostModel::superconducting_default();
let energy = EnergyModel::a100();

// ── VQE loop ──
let evals_per_iter = gradient.circuit_evaluations(num_params);
let time_per_eval_us = qcm.circuit_time_us(10, 5, 1, 8);
let fidelity_per_eval = qcm.circuit_fidelity(10, 5);

for iter in 0..100 {
    let iter_time_ms = (evals_per_iter as f64 * time_per_eval_us) / 1000.0;
    tracker.consume(0, 0, iter_time_ms, fidelity_per_eval);

    if let Err(e) = tracker.check_remaining() {
        println!("Stopped at iteration {}: {}", iter, e);
        break;
    }
}

println!("Total time: {:.2} ms", tracker.elapsed_ms);
println!("Final fidelity: {:.6}", tracker.current_fidelity);
let total_energy_j = energy.quantum_energy_joules(
    tracker.elapsed_ms * 1000.0, encoding.num_qubits);
println!("Energy: {:.2} J", total_energy_j);
```

---

## 18. Configuration with `.lith` Files

### 18.1 Overview

LIFT uses `.lith` configuration files to control the compilation pipeline. The format is INI-like with `[section]` headers and `key = value` pairs. Comments use `#` or `//`.

### 18.2 Full `.lith` Example

```ini
# my_project.lith — LIFT compilation configuration

[target]
backend = "llvm"        # llvm | qasm
device = "H100"         # A100 | H100
precision = "fp16"      # fp32 | fp16 | bf16 | fp8 | int8

[budget]
max_flops = 1000000000000    # 1 TFLOP
max_memory_bytes = 80000000000  # 80 GB
max_time_ms = 100.0          # 100 ms
min_fidelity = 0.95          # 95% quantum fidelity

[optimisation]
level = O3                   # O0 | O1 | O2 | O3
max_iterations = 20

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = "heavy_hex"
num_qubits = 127
shots = 4096
error_mitigation = "zne"     # zero-noise extrapolation
```

### 18.3 Configuration Sections

| Section | Keys | Description |
|---------|------|-------------|
| **[target]** | `backend`, `device`, `precision` | Compilation target |
| **[budget]** | `max_flops`, `max_memory_bytes`, `max_time_ms`, `min_fidelity`, `max_circuit_depth` | Resource constraints |
| **[optimisation]** | `level`, `passes`, `disabled_passes`, `max_iterations` | Pass pipeline control |
| **[simulation]** | `shape_propagation`, `flop_counting`, `memory_analysis`, `noise_simulation` | Analysis toggles |
| **[quantum]** | `topology`, `num_qubits`, `error_mitigation`, `shots` | Quantum device settings |

### 18.4 Optimisation Levels

| Level | Passes |
|-------|--------|
| **O0** | No optimisation |
| **O1** | Canonicalize, DCE |
| **O2** | Canonicalize, constant folding, DCE, tensor fusion (default) |
| **O3** | All passes including FlashAttention, CSE, gate cancellation, rotation merge |

### 18.5 Loading Configuration Programmatically

```rust
use lift_config::{ConfigParser, LithConfig};

// From .lith file
let source = std::fs::read_to_string("project.lith").unwrap();
let config = ConfigParser::new().parse(&source).expect("Config parse error");

println!("Backend: {}", config.target.backend);
println!("Opt level: {:?}", config.optimisation.level);

if let Some(q) = &config.quantum {
    println!("Quantum: {} qubits, {} topology", q.num_qubits, q.topology);
}

// From JSON
let json = r#"{"target":{"backend":"qasm","device":null,"precision":"fp32"},
               "budget":{"max_flops":null,"max_memory_bytes":null,
                         "max_time_ms":null,"min_fidelity":0.99,
                         "max_circuit_depth":null},
               "optimisation":{"level":"O2","passes":["canonicalize","dce"],
                               "disabled_passes":[],"max_iterations":10},
               "simulation":{"enable_shape_propagation":true,
                             "enable_flop_counting":true,
                             "enable_memory_analysis":true,
                             "enable_noise_simulation":true},
               "quantum":{"topology":"grid","num_qubits":27,
                          "error_mitigation":null,"shots":4096}}"#;

let config = ConfigParser::new().parse_json(json).expect("JSON parse error");
```

### 18.6 Default Configuration

When no `.lith` file is provided, LIFT uses these defaults:

```rust
let config = LithConfig::default();
// target.backend = "llvm"
// target.precision = "fp32"
// optimisation.level = O2
// optimisation.passes = ["canonicalize", "constant-folding", "dce", "tensor-fusion"]
// simulation: all enabled
// quantum: None
```

---

## 19. CLI Reference

### 19.1 Installation

After building with `cargo build --release`, the CLI binary is at `target/release/lift`.

### 19.2 Commands

#### `lift verify` — Verify a `.lif` file

Checks SSA invariants, type correctness, and qubit linearity.

```bash
lift verify examples/tensor_mlp.lif
```

Output:
```
Verification passed: examples/tensor_mlp.lif
  Values: 11
  Operations: 6
  Blocks: 1
  Regions: 1
```

```bash
# Verbose mode
lift -v verify examples/quantum_bell.lif
```

#### `lift analyse` — Analyse resource usage

Computes FLOPs, memory, gate counts, and fidelity estimates.

```bash
lift analyse examples/tensor_mlp.lif
```

Output:
```
=== LIFT Analysis Report ===
File: examples/tensor_mlp.lif

Operations: 6
  Tensor ops: 6
  Quantum ops: 0
  Hybrid ops: 0

Compute:
  Total FLOPs: 803.33 KFLOP
  Total memory: 3.10 MiB
  Peak memory: 3.10 MiB

Op breakdown:
  tensor.matmul: 2
  tensor.add: 2
  tensor.relu: 1
  tensor.softmax: 1
```

JSON output:
```bash
lift analyse examples/tensor_mlp.lif --format json
```

#### `lift print` — Print human-readable IR

```bash
lift print examples/quantum_bell.lif
```

Output:
```
module @bell_state {
    func @bell(%v0: qubit, %v1: qubit) -> (qubit, qubit) {
    ^bb0(%v0: qubit, %v1: qubit):
        %v2 = "quantum.h"(%v0) : (qubit) -> qubit
        %v3, %v4 = "quantum.cx"(%v2, %v1) : (qubit, qubit) -> (qubit, qubit)
    }
}
```

#### `lift optimise` — Run optimisation passes

```bash
# Default passes (O2)
lift optimise examples/tensor_mlp.lif -o optimised.lif

# With custom config
lift optimise examples/tensor_mlp.lif --config project.lith -o optimised.lif
```

Output:
```
Optimisation results:
  canonicalize -> unchanged
  constant-folding -> unchanged
  dce -> unchanged
  tensor-fusion -> changed
Output written to: optimised.lif
```

#### `lift predict` — Predict performance

```bash
# Predict on A100 (default)
lift predict examples/tensor_mlp.lif

# Predict on H100
lift predict examples/tensor_mlp.lif --device h100
```

Output:
```
=== LIFT Performance Prediction ===
Device: H100

Compute time: 0.0000 ms
Memory time: 0.0009 ms
Predicted time: 0.0009 ms
Arithmetic intensity: 266.67 FLOP/byte
Bottleneck: memory
```

#### `lift export` — Export to backend

```bash
# Export to LLVM IR
lift export examples/tensor_mlp.lif --backend llvm -o model.ll

# Export to OpenQASM 3.0
lift export examples/quantum_bell.lif --backend qasm -o circuit.qasm

# Print to stdout
lift export examples/quantum_bell.lif --backend qasm
```

### 19.3 Global Flags

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Enable debug-level logging |
| `--version` | Print version |
| `--help` | Print help |

---

## 20. Complete API Reference

### 20.1 Crate Overview

| Crate | Purpose |
|-------|---------|
| **lift-core** | IR foundation: Context, types, values, operations, blocks, regions, verifier, printer, pass manager |
| **lift-ast** | Lexer, parser, IR builder for `.lif` files |
| **lift-tensor** | Tensor operations (107), shape inference, FLOPs computation |
| **lift-quantum** | Quantum gates (46+), noise models, topology, QEC codes, Kraus channels |
| **lift-hybrid** | Hybrid operations (21), encoding strategies, gradient methods |
| **lift-opt** | Optimisation passes (11): canonicalize, fusion, FlashAttention, gate cancellation, etc. |
| **lift-sim** | Cost models (GPU + QPU), analysis reports, energy models, budgets |
| **lift-predict** | Roofline prediction (classical), quantum prediction (fidelity + shots) |
| **lift-import** | Importers: ONNX, PyTorch FX, OpenQASM 3.0 |
| **lift-export** | Exporters: LLVM IR, OpenQASM 3.0 |
| **lift-config** | `.lith` configuration parser |
| **lift-cli** | Command-line interface |

### 20.2 lift-core API

**Context** — central IR container:

| Method | Description |
|--------|-------------|
| `Context::new()` | Create empty IR context |
| `ctx.intern_string(s)` → `StringId` | Intern a string |
| `ctx.resolve_string(id)` → `&str` | Resolve interned string |
| `ctx.intern_type(ty)` → `TypeId` | Intern a type |
| `ctx.resolve_type(id)` → `&CoreType` | Resolve interned type |
| `ctx.make_integer_type(bits, signed)` → `TypeId` | Create integer type |
| `ctx.make_float_type(bits)` → `TypeId` | Create float type |
| `ctx.make_boolean_type()` → `TypeId` | Create boolean type |
| `ctx.make_tensor_type(shape, dtype, layout)` → `TypeId` | Create tensor type |
| `ctx.make_qubit_type()` → `TypeId` | Create qubit type |
| `ctx.make_bit_type()` → `TypeId` | Create classical bit type |
| `ctx.make_void_type()` → `TypeId` | Create void type |
| `ctx.make_index_type()` → `TypeId` | Create index type |
| `ctx.create_block()` → `BlockKey` | Create a new block |
| `ctx.create_block_arg(block, ty)` → `ValueKey` | Add block argument |
| `ctx.create_op(name, dialect, inputs, types, attrs, loc)` → `(OpKey, Vec<ValueKey>)` | Create operation |
| `ctx.add_op_to_block(block, op)` | Add op to block |
| `ctx.create_region()` → `RegionKey` | Create a region |
| `ctx.create_module(name)` → `usize` | Create a module |
| `ctx.snapshot()` | Snapshot context state |

**Verifier:**

| Function | Description |
|----------|-------------|
| `verifier::verify(&ctx)` → `Result<(), Vec<VerifyError>>` | Verify the full IR |

**Printer:**

| Function | Description |
|----------|-------------|
| `printer::print_ir(&ctx)` → `String` | Print IR as text |

**Pass Manager:**

| Method | Description |
|--------|-------------|
| `PassManager::new()` | Create pass manager |
| `pm.add_pass(Box<dyn Pass>)` | Register a pass |
| `pm.run_all(&mut ctx)` → `Vec<(String, PassResult)>` | Run all passes |

**Types:**

| Type | Variants |
|------|----------|
| `CoreType` | `Integer`, `Float`, `Boolean`, `Tuple`, `Function`, `Opaque`, `Void`, `Index` |
| `TypeData` | `None`, `Tensor(TensorTypeInfo)`, `Qubit`, `ClassicalBit`, `Hamiltonian`, `QuantumState` |
| `DataType` | `FP32`, `FP16`, `BF16`, `FP64`, `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `Bool` |
| `Dimension` | `Constant(usize)`, `Dynamic` |
| `MemoryLayout` | `Contiguous`, `Strided`, `Blocked` |

**Attributes:**

| Type | Variants |
|------|----------|
| `Attribute` | `Integer(i64)`, `Float(f64)`, `String(StringId)`, `Bool(bool)`, `Type(TypeId)`, `Array(Vec)`, `Dict(HashMap)` |
| `Attributes` | `.set(key, attr)`, `.get(key)`, `.get_integer(key)`, `.get_float(key)`, `.get_bool(key)` |

### 20.3 lift-tensor API

| Item | Description |
|------|-------------|
| `TensorOp` enum | 107 tensor operations |
| `TensorOp::name()` → `&str` | Get string name |
| `TensorOp::from_name(s)` → `Option<TensorOp>` | Parse from string |
| `TensorOp::num_inputs()` → `(usize, usize)` | Min/max input count |
| `TensorOp::flops_formula()` → `&str` | Theoretical FLOPs formula |
| `TensorOp::is_zero_flop()` → `bool` | True for shape-only ops |
| `TensorOp::is_activation()` → `bool` | True for activation ops |
| `TensorOp::is_attention()` → `bool` | True for attention variants |
| `TensorOp::is_convolution()` → `bool` | True for conv ops |
| `TensorOp::is_fused()` → `bool` | True for fused kernels |
| `TensorOp::is_gradient()` → `bool` | True for gradient ops |
| `ShapeInference::infer_output_shape(op, inputs)` → `Result<Vec<TensorTypeInfo>>` | Infer output shapes |
| `ShapeInference::compute_flops(op, inputs)` → `Option<u64>` | Count FLOPs |
| `ShapeInference::compute_memory_bytes(op, inputs)` → `Option<u64>` | Estimate memory |

### 20.4 lift-quantum API

| Item | Description |
|------|-------------|
| `QuantumGate` enum | 46+ quantum gates |
| `QuantumGate::op_name()` → `&str` | Get gate name (e.g. `"quantum.h"`) |
| `QuantumGate::from_name(s)` → `Option<QuantumGate>` | Parse from string |
| `QuantumGate::num_qubits()` → `usize` | Gate arity |
| `QuantumGate::is_parametric()` → `bool` | Requires angle parameters |
| `QuantumGate::is_self_inverse()` → `bool` | G·G = I |
| `QuantumGate::is_clifford()` → `bool` | In Clifford group |
| `QuantumGate::is_measurement()` → `bool` | Measurement or control |
| `QuantumGate::is_entangling()` → `bool` | Creates entanglement |
| `QuantumGate::native_basis(provider)` → `&[QuantumGate]` | Hardware-native gates |
| `Provider` enum | `IbmEagle`, `IbmKyoto`, `Rigetti`, `IonQ`, `Quantinuum`, `Generic` |
| `NoiseModel` enum | `Ideal`, `Depolarizing`, `AmplitudeDamping`, `PhaseDamping`, `BitFlip`, `PhaseFlip`, `ThermalRelaxation`, `Kraus`, `Composed` |
| `NoiseModel::fidelity()` → `f64` | Compute fidelity |
| `NoiseModel::compose(other)` → `NoiseModel` | Chain noise models |
| `GateNoise::ideal()` | Perfect gate |
| `GateNoise::with_depolarizing(f, t)` | Gate with depolarizing noise |
| `CircuitNoise::new()` | Track circuit-level noise |
| `CircuitNoise::add_gate(noise, is_2q)` | Add gate to circuit |
| `CircuitNoise::meets_threshold(min)` → `bool` | Check fidelity threshold |
| `DeviceTopology::linear(n)` | Linear chain |
| `DeviceTopology::grid(r, c)` | 2D grid |
| `DeviceTopology::heavy_hex(n)` | IBM heavy-hex |
| `DeviceTopology::all_to_all(n)` | Full connectivity |
| `DeviceTopology::tree(n)` | Binary tree |
| `DeviceTopology::custom(name, edges, fid)` | Custom topology |
| `topo.are_connected(q0, q1)` → `bool` | Check edge |
| `topo.neighbors(q)` → `Vec<usize>` | Get neighbours |
| `topo.shortest_path(from, to)` → `Option<Vec<usize>>` | BFS path |
| `topo.swap_distance(from, to)` → `Option<usize>` | SWAP count |
| `topo.diameter()` → `usize` | Graph diameter |
| `topo.avg_connectivity()` → `f64` | Average degree |

### 20.5 lift-hybrid API

| Item | Description |
|------|-------------|
| `HybridOp` enum | 21 hybrid operations |
| `HybridOp::op_name()` → `&str` | Get op name |
| `HybridOp::from_name(s)` → `Option<HybridOp>` | Parse from string |
| `HybridOp::is_gradient()` → `bool` | Gradient op? |
| `HybridOp::is_variational()` → `bool` | Variational algorithm? |
| `EncodingStrategy` enum | `AngleEncoding`, `AmplitudeEncoding`, `BasisEncoding`, `IQPEncoding`, `HamiltonianEncoding`, `KernelEncoding` |
| `EncodingStrategy::qubits_required(dim)` → `usize` | Qubits needed |
| `EncodingStrategy::circuit_depth(dim)` → `usize` | Circuit depth |
| `EncodingConfig::new(strategy, dim)` | Create config |
| `GradientMethod` enum | `ParameterShift`, `FiniteDifference`, `SPSA`, `Adjoint`, `Backprop` |
| `GradientMethod::circuit_evaluations(n)` → `usize` | Evaluations needed |
| `GradientMethod::is_exact()` → `bool` | Exact gradient? |
| `JointGradientConfig` | Combined classical+quantum gradients |
| `JointGradientConfig::total_evaluations()` → `usize` | Total eval count |
| `AnsatzType` enum | `HardwareEfficient`, `StronglyEntangling`, `TwoLocal`, `UCCSD`, `Custom` |
| `SyncPolicy` enum | `Blocking`, `Asynchronous`, `Pipeline` |
| `FeatureMap` enum | `ZZFeatureMap`, `PauliFeatureMap`, `AngleEncoding`, `AmplitudeEncoding` |

### 20.6 lift-opt Passes

| Pass | Name | Description |
|------|------|-------------|
| `Canonicalize` | `"canonicalize"` | Simplify: x+0→x, x×1→x, reshape(reshape(x))→reshape(x) |
| `ConstantFolding` | `"constant-folding"` | Evaluate constant expressions at compile time |
| `DeadCodeElimination` | `"dce"` | Remove unused operations |
| `TensorFusion` | `"tensor-fusion"` | Fuse matmul+bias+relu into single kernel |
| `GateCancellation` | `"gate-cancellation"` | Cancel adjacent inverse gates (H·H→I) |
| `RotationMerge` | `"rotation-merge"` | Merge rotations: Rz(a)·Rz(b)→Rz(a+b) |
| `FlashAttentionPass` | `"flash-attention"` | Replace attention with FlashAttention when seq_len > threshold |
| `CommonSubexprElimination` | `"cse"` | Eliminate duplicate computations |
| `QuantisationPass` | `"quantisation"` | Annotate quantisable operations |
| `NoiseAwareSchedule` | `"noise-aware-schedule"` | Reorder gates for minimal noise |
| `LayoutMapping` | `"layout-mapping"` | Map logical qubits to physical topology |

### 20.7 lift-sim API

| Item | Description |
|------|-------------|
| `CostModel::a100()` | NVIDIA A100 profile (312 TFLOPS, 2039 GB/s, 80GB) |
| `CostModel::h100()` | NVIDIA H100 profile (989 TFLOPS, 3350 GB/s, 80GB) |
| `model.compute_time_ms(flops)` → `f64` | Compute-only time |
| `model.memory_time_ms(bytes)` → `f64` | Memory-only time |
| `model.roofline_time_ms(flops, bytes)` → `f64` | Roofline prediction |
| `model.arithmetic_intensity(flops, bytes)` → `f64` | FLOP/byte ratio |
| `model.is_compute_bound(flops, bytes)` → `bool` | Compute or memory bound |
| `model.fits_in_memory(bytes)` → `bool` | Fits in GPU VRAM |
| `model.num_gpus_needed(bytes)` → `usize` | GPUs required |
| `QuantumCostModel::superconducting_default()` | IBM-like QPU |
| `QuantumCostModel::trapped_ion_default()` | IonQ-like QPU |
| `QuantumCostModel::neutral_atom_default()` | Neutral-atom QPU |
| `qcm.circuit_fidelity(n_1q, n_2q)` → `f64` | Gate fidelity product |
| `qcm.circuit_time_us(n_1q, n_2q, n_meas, depth)` → `f64` | Execution time |
| `qcm.decoherence_fidelity(time_us)` → `f64` | Decoherence fidelity |
| `EnergyModel::a100()` / `::h100()` | Energy profiles |
| `energy.energy_joules(time_ms, gpus)` → `f64` | Energy in joules |
| `energy.energy_kwh(time_ms, gpus)` → `f64` | Energy in kWh |
| `energy.carbon_grams(time_ms, gpus)` → `f64` | CO₂ in grams |
| `energy.quantum_energy_joules(time_us, qubits)` → `f64` | Quantum energy |
| `Budget` struct | Static resource constraints |
| `budget.check_flops(n)` / `check_memory(n)` / `check_fidelity(f)` | Constraint checks |
| `ReactiveBudget::new(budget)` | Dynamic budget tracker |
| `tracker.consume(flops, mem, time, fidelity)` | Record usage |
| `tracker.check_remaining()` → `Result<()>` | Check all constraints |
| `tracker.remaining_flops()` / `remaining_time_ms()` | Remaining budget |
| `tracker.utilisation()` → `BudgetUtilisation` | Usage ratios |
| `analyze_module(&ctx)` → `AnalysisReport` | Full module analysis |
| `analyze_block(&ctx, block)` → `AnalysisReport` | Single block analysis |

### 20.8 lift-predict API

| Item | Description |
|------|-------------|
| `predict_performance(report, cost_model)` → `RooflineResult` | Classical roofline prediction |
| `predict_quantum(analysis, qcm, precision)` → `QuantumPrediction` | Quantum performance prediction |
| `RooflineResult` | `.compute_time_ms`, `.memory_time_ms`, `.predicted_time_ms`, `.arithmetic_intensity`, `.is_compute_bound`, `.bottleneck` |
| `QuantumPrediction` | `.estimated_fidelity`, `.circuit_time_us`, `.num_shots_for_precision`, `.total_execution_time_ms` |

---

## 21. Troubleshooting

### 21.1 Common Verification Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `SSA violation: value used but not defined` | Using a `%name` that was never created | Ensure all operands are defined before use |
| `SSA violation: value defined more than once` | Two operations produce the same value | Use unique result names |
| `Dominance violation` | Using a value before its defining op in block order | Reorder operations so definitions come before uses |
| `Type mismatch` | Input types don't match operation signature | Check tensor shapes and data types |
| `Linearity violation: qubit consumed more than once` | A qubit value used as input to two operations | Each qubit must be consumed exactly once |
| `Linearity violation: qubit not consumed (leaked)` | A qubit is created but never used | Ensure all qubits are measured or returned |
| `Missing terminator` | A block has no `return` or `branch` at the end | Add a terminator operation |

### 21.2 Common Parse Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Unexpected token | Syntax error in `.lif` file | Check operation format: `%r = "dialect.op"(%args) : (types) -> type` |
| Unknown type | Type name not recognised | Use `tensor<...>`, `qubit`, `bit`, `f32`, `i64`, `bool` |
| Unresolved dialect | Using an op without declaring the dialect | Add `#dialect tensor`, `#dialect quantum`, or `#dialect hybrid` at file top |

### 21.3 Optimisation Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Fusion not applied | Pattern not matched (e.g. different order) | Ensure `matmul → add → relu` pattern is present |
| FlashAttention not applied | `seq_len` attribute missing or below threshold | Set `seq_len` attribute on attention ops, or lower threshold |
| Pass returns Error | IR is in invalid state | Run `verify` before optimisation |

### 21.4 Performance Debugging

```rust
// Check if compute-bound or memory-bound
let model = CostModel::a100();
let report = analyze_module(&ctx);

if model.is_compute_bound(report.total_flops, report.total_memory_bytes) {
    println!("Compute-bound: reduce FLOPs (quantise, prune, fuse)");
} else {
    println!("Memory-bound: reduce data movement (fusion, recomputation)");
}

// Check per-op breakdown
for (op, count) in &report.op_breakdown {
    println!("  {}: {} instances", op, count);
}
```

### 21.5 Quantum Debugging

```rust
use lift_quantum::{CircuitNoise, GateNoise};

// Track where fidelity drops
let mut circuit = CircuitNoise::new();
let g1q = GateNoise::with_depolarizing(0.999, 0.02);
let g2q = GateNoise::with_depolarizing(0.99, 0.3);

// After each gate, check fidelity
circuit.add_gate(&g1q, false);
println!("After H: fidelity = {:.6}", circuit.total_fidelity);

circuit.add_gate(&g2q, true);
println!("After CX: fidelity = {:.6}", circuit.total_fidelity);

// 2Q gates dominate fidelity loss!
```

---

## Appendix A — Summary of All Operations

| Dialect | Count | Categories |
|---------|-------|-----------|
| **tensor** | 107 | Arithmetic, activations, normalisation, shape, attention, convolution, pooling, recurrent, math, sparse, quantisation, diffusion, GNN, memory, gradient, parallelism, fused |
| **quantum** | 46+ | 1Q standard, 1Q parametric, 1Q fixed, 2Q standard, 2Q parametric, IonQ native, 3Q, multi-controlled, measurement, special |
| **hybrid** | 21 | Encoding, gradient methods, processing, variational, data transfer, co-execution, measurement |

**Total: 174+ operations** across three dialects in a single unified IR.

---

## Appendix B — Quick Reference Card

```
# Parse and verify
lift verify input.lif

# Analyse (FLOPs, memory, gates)
lift analyse input.lif

# Optimise with default passes
lift optimise input.lif -o optimised.lif

# Optimise with config
lift optimise input.lif --config project.lith -o optimised.lif

# Predict performance on H100
lift predict input.lif --device h100

# Export to LLVM
lift export input.lif --backend llvm -o model.ll

# Export to OpenQASM
lift export input.lif --backend qasm -o circuit.qasm
```

---

*LIFT v0.2.1 — MIT License — https://github.com/rustnew/Lift*
