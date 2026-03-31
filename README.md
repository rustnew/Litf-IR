# Complete LIFT Framework Guide — All Features

> **LIFT** — *Language for Intelligent Frameworks and Technologies*
> Unified intermediate representation for AI and quantum computing.

This document describes **every feature** of the LIFT framework, numbered and organised by crate. For each feature: what it does, how to use it, and which other features to combine it with.

---

## Table of Contents

1. [General Architecture](#1-general-architecture)
2. [lift-core — IR Core](#2-lift-core--ir-core)
3. [lift-ast — Parsing the .lif Language](#3-lift-ast--parsing-the-lif-language)
4. [lift-tensor — Tensor Operations (90+ ops)](#4-lift-tensor--tensor-operations-90-ops)
5. [lift-quantum — Quantum Gates and Noise (50+ gates)](#5-lift-quantum--quantum-gates-and-noise-50-gates)
6. [lift-hybrid — Classical-Quantum Hybrid Computation](#6-lift-hybrid--classical-quantum-hybrid-computation)
7. [lift-opt — Optimisation Passes (11 passes)](#7-lift-opt--optimisation-passes-11-passes)
8. [lift-sim — Simulation and Cost Analysis](#8-lift-sim--simulation-and-cost-analysis)
9. [lift-predict — Performance Prediction](#9-lift-predict--performance-prediction)
10. [lift-import — Model Import](#10-lift-import--model-import)
11. [lift-export — Backend Export](#11-lift-export--backend-export)
12. [lift-config — Configuration (.lith)](#12-lift-config--configuration-lith)
13. [lift-cli — Command-Line Interface](#13-lift-cli--command-line-interface)
14. [Combinations and Complete Pipelines](#14-combinations-and-complete-pipelines)
15. [Concrete Examples](#15-concrete-examples)

---

## 1. General Architecture

LIFT is a modular compiler composed of **13 crates** organised in layers:

```
                    ┌──────────┐
                    │ lift-cli │  ← User interface
                    └────┬─────┘
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │ lift-import │ │lift-opt │ │ lift-export │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
           │             │             │
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │  lift-ast   │ │lift-sim │ │lift-predict │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
           │             │             │
    ┌──────┴─────────────┴─────────────┴──────┐
    │              lift-core                    │
    ├──────────┬──────────┬───────────────────┤
    │lift-tensor│lift-quantum│  lift-hybrid    │
    └──────────┴──────────┴───────────────────┘
```

### 1.1 Compilation Pipeline

The standard workflow is:

```
Source (.lif) → Lexer → Parser → IR (SSA) → Verification → Optimisation → Simulation → Export
```

### 1.2 File Formats

| Extension | Description |
|-----------|-------------|
| `.lif`    | LIFT IR source code |
| `.lith`   | Compilation configuration |

### 1.3 Adding LIFT as a Dependency

```toml
[dependencies]
lift-core     = "0.2.0"
lift-tensor   = "0.2.0"
lift-quantum  = "0.2.0"
lift-hybrid   = "0.2.0"
lift-opt      = "0.2.0"
lift-sim      = "0.2.0"
lift-predict  = "0.2.0"
lift-import   = "0.2.0"
lift-export   = "0.2.0"
lift-config   = "0.2.0"
```

---

## 2. lift-core — IR Core

The heart of the framework. Provides the SSA (Static Single Assignment) intermediate representation.

### 2.1 Context — The Central Container

```rust
use lift_core::Context;

let mut ctx = Context::new();
```

The `Context` stores **all** IR data: values, operations, blocks, regions, functions, modules, interned strings, and types.

| Field | Description | Usage |
|-------|-------------|-------|
| `ctx.values` | All SSA values | Each operation result is a unique value |
| `ctx.ops` | All operations | Program instructions |
| `ctx.blocks` | Basic blocks | Contain sequences of operations |
| `ctx.regions` | Regions | Contain blocks (function bodies) |
| `ctx.modules` | Modules | Compilation units |
| `ctx.strings` | String interning | `ctx.strings.intern("name")` |
| `ctx.type_interner` | Type interning | Type deduplication |

**Combine with**: All other crates. The `Context` is the entry point for every pipeline.

### 2.2 Types — Type System

```rust
use lift_core::types::*;

// Data types
let fp32 = DataType::FP32;
let fp16 = DataType::FP16;
let bf16 = DataType::BF16;
let int8 = DataType::INT8;
let fp64 = DataType::FP64;

// Dimensions (static or symbolic)
let batch = Dimension::Constant(32);
let seq = Dimension::Symbolic("seq_len".to_string());

// Tensor type info
let tensor_info = TensorTypeInfo {
    shape: vec![Dimension::Constant(1), Dimension::Constant(784)],
    dtype: DataType::FP32,
    layout: MemoryLayout::Contiguous,
};

// Size in bytes
let bytes = tensor_info.size_bytes(); // Some(3136) = 1*784*4
```

**Available data types**:

| Type | Size | Usage |
|------|------|-------|
| `FP64` | 8 bytes | High-precision scientific computing |
| `FP32` | 4 bytes | Standard training |
| `FP16` | 2 bytes | Fast inference |
| `BF16` | 2 bytes | Mixed-precision training (Google Brain) |
| `INT8` | 1 byte | Post-training quantisation |
| `INT32` | 4 bytes | Indices, counters |
| `BOOL` | 1 byte | Masks |

**Memory layouts**: `Contiguous`, `Strided`.

### 2.3 Attributes — Operation Metadata

```rust
use lift_core::attributes::{Attribute, Attributes};

let mut attrs = Attributes::new();

// Different attribute types
attrs.set("num_heads", Attribute::Integer(8));
attrs.set("dropout", Attribute::Float(0.1));
attrs.set("causal", Attribute::Bool(true));

// Reading
let heads = attrs.get_integer("num_heads"); // Some(8)
let drop = attrs.get_float("dropout");       // Some(0.1)
let causal = attrs.get_bool("causal");       // Some(true)

// Checking
assert!(attrs.contains("num_heads"));
assert_eq!(attrs.len(), 3);

// Iteration
for (key, val) in attrs.iter() {
    println!("{}: {:?}", key, val);
}
```

**Combine with**: `lift-opt` (passes read/write attributes), `lift-export` (exporters read attributes).

### 2.4 Verifier — Invariant Checking

```rust
use lift_core::verifier;

let ctx = Context::new();
match verifier::verify(&ctx) {
    Ok(()) => println!("IR valid"),
    Err(errors) => {
        for e in &errors {
            eprintln!("Error: {}", e);
        }
    }
}
```

Checks:
- **SSA**: every value is defined exactly once
- **Qubit linearity**: every qubit is used exactly once
- **Typing**: type consistency between operations
- **Structure**: blocks, regions, terminators are correct

**Combine with**: Always use after import and after each optimisation pass.

### 2.5 Printer — IR Display

```rust
use lift_core::printer::print_ir;

let ctx = Context::new();
let output = print_ir(&ctx);
println!("{}", output);
```

Produces a human-readable textual representation of the IR, useful for debugging.

### 2.6 Pass Manager

```rust
use lift_core::pass::{PassManager, Pass, PassResult, AnalysisCache};

let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.add_pass(Box::new(lift_opt::TensorFusion));

let results = pm.run_all(&mut ctx);
for (name, result) in &results {
    match result {
        PassResult::Changed => println!("{}: changed", name),
        PassResult::Unchanged => println!("{}: unchanged", name),
        PassResult::Error(e) => println!("{}: error: {}", name, e),
        PassResult::RolledBack => println!("{}: rolled back", name),
    }
}
```

**Combine with**: `lift-opt` (all 11 passes), `lift-config` (pass selection via configuration).

### 2.7 Dialect — Dialect System

```rust
use lift_core::dialect::{DialectRegistry, Dialect};

let registry = DialectRegistry::new();
// The tensor, quantum, hybrid dialects are registered automatically
```

The three LIFT dialects:
- **tensor**: tensor operations (`tensor.matmul`, `tensor.relu`, etc.)
- **quantum**: quantum gates (`quantum.h`, `quantum.cx`, etc.)
- **hybrid**: hybrid operations (`hybrid.encode`, `hybrid.vqc_layer`, etc.)

---

## 3. lift-ast — Parsing the .lif Language

### 3.1 Lexer — Tokenisation

```rust
use lift_ast::Lexer;

let source = r#"
#dialect tensor
module @mlp {
    func @forward(%x: tensor<1x784xf32>) -> tensor<1x10xf32> {
        %out = "tensor.relu"(%x) : (tensor<1x784xf32>) -> tensor<1x784xf32>
        return %out
    }
}
"#;

let mut lexer = Lexer::new(source);
let tokens = lexer.tokenize().to_vec();
assert!(lexer.errors().is_empty(), "Lexing errors: {:?}", lexer.errors());
```

### 3.2 Parser — Syntactic Analysis

```rust
use lift_ast::Parser;

let mut parser = Parser::new(tokens);
let program = parser.parse().expect("Parsing errors");
```

### 3.3 IrBuilder — IR Construction

```rust
use lift_ast::IrBuilder;
use lift_core::Context;

let mut ctx = Context::new();
let mut builder = IrBuilder::new();
builder.build_program(&mut ctx, &program).expect("IR construction errors");
```

### 3.4 Complete Parsing Pipeline

```rust
fn load_lif_file(path: &str) -> Result<Context, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Read failed: {}", e))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().to_vec();
    if !lexer.errors().is_empty() {
        return Err(format!("Lexer errors: {:?}", lexer.errors()));
    }

    let mut parser = Parser::new(tokens);
    let program = parser.parse().map_err(|e| format!("Parser errors: {:?}", e))?;

    let mut ctx = Context::new();
    let mut builder = IrBuilder::new();
    builder.build_program(&mut ctx, &program)?;
    Ok(ctx)
}
```

**Combine with**: `lift-core` (Context), then `lift-opt` (optimisation), `lift-sim` (analysis), `lift-export` (compilation).

---

## 4. lift-tensor — Tensor Operations (90+ ops)

### 4.1 Complete Operation List by Category

#### 4.1.1 Basic Arithmetic (5 ops)

| # | Op | IR Name | Inputs | Description |
|---|-----|---------|--------|-------------|
| 1 | `Add` | `tensor.add` | 2 | Element-wise addition |
| 2 | `Sub` | `tensor.sub` | 2 | Subtraction |
| 3 | `Mul` | `tensor.mul` | 2 | Element-wise multiplication |
| 4 | `Div` | `tensor.div` | 2 | Division |
| 5 | `Neg` | `tensor.neg` | 1 | Negation |

```rust
use lift_tensor::ops::TensorOp;

let op = TensorOp::MatMul;
println!("Nom: {}", op.name());           // "tensor.matmul"
println!("Inputs: {:?}", op.num_inputs()); // (2, 2)
println!("FLOPs: {}", op.flops_formula()); // "2*M*N*K"
```

#### 4.1.2 Linear Algebra (4 ops)

| # | Op | Inputs | Description |
|---|-----|--------|-------------|
| 6 | `MatMul` | 2 | Matrix multiplication |
| 7 | `Linear` | 3 | Linear layer (matmul + bias) |
| 8 | `Embedding` | 2 | Embedding lookup table |
| 9 | `SparseMatMul` | 2 | Sparse MatMul |

#### 4.1.3 Activations (11 ops)

| # | Op | Description | FLOPs Formula |
|---|-----|-------------|---------------|
| 10 | `ReLU` | max(0, x) | N |
| 11 | `GeLU` | Gaussian Error Linear Unit | ~8N |
| 12 | `SiLU` | x * sigmoid(x) (Swish) | ~8N |
| 13 | `Sigmoid` | 1/(1+exp(-x)) | N |
| 14 | `Tanh` | Hyperbolic tangent | N |
| 15 | `Softmax` | exp(x)/sum(exp(x)) | 5N |
| 16 | `LeakyReLU` | max(αx, x) | N |
| 17 | `ELU` | Exponential Linear Unit | N |
| 18 | `Mish` | x * tanh(softplus(x)) | ~8N |
| 19 | `HardSwish` | Swish approximation | ~8N |
| 20 | `HardSigmoid` | Sigmoid approximation | N |

```rust
assert!(TensorOp::ReLU.is_activation());
assert!(!TensorOp::MatMul.is_activation());
```

#### 4.1.4 Normalisation (5 ops)

| # | Op | Inputs | Description |
|---|-----|--------|-------------|
| 21 | `LayerNorm` | 2-3 | Layer normalisation |
| 22 | `RMSNorm` | 2-3 | Root Mean Square Norm (LLaMA) |
| 23 | `BatchNorm` | 3-5 | Batch normalisation |
| 24 | `GroupNorm` | 2-3 | Group normalisation |
| 25 | `InstanceNorm` | 2-3 | Instance normalisation |

```rust
assert!(TensorOp::LayerNorm.is_normalisation());
```

#### 4.1.5 Attention (8 ops)

| # | Op | Inputs | Description |
|---|-----|--------|-------------|
| 26 | `Attention` | 3-4 | Standard attention (Q, K, V, [mask]) |
| 27 | `MultiHeadAttention` | 3-4 | Multi-head |
| 28 | `MultiQueryAttention` | 3-4 | Multi-query (Llama) |
| 29 | `GroupedQueryAttention` | 3-4 | Grouped query (GQA) |
| 30 | `FlashAttention` | 3-4 | FlashAttention V2 (O(N) memory) |
| 31 | `SlidingWindowAttention` | 3-4 | Sliding window (Mistral) |
| 32 | `CrossAttention` | 3-4 | Cross-attention (encoder-decoder) |
| 33 | `PagedAttention` | 3-5 | Paged attention (vLLM) |

```rust
assert!(TensorOp::FlashAttention.is_attention());
```

#### 4.1.6 Convolutions (6 ops)

| # | Op | Description |
|---|-----|-------------|
| 34 | `Conv2D` | Convolution 2D standard |
| 35 | `Conv1D` | 1D convolution (audio, sequences) |
| 36 | `Conv3D` | 3D convolution (video, volumetric) |
| 37 | `ConvTranspose2D` | Transposed convolution (upsampling) |
| 38 | `DepthwiseConv2D` | Depthwise convolution (MobileNet) |
| 39 | `DilatedConv2D` | Dilated convolution (large receptive field) |

#### 4.1.7 Pooling (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 40 | `MaxPool2D` | Max pooling 2D |
| 41 | `AvgPool2D` | Average pooling 2D |
| 42 | `AdaptiveAvgPool2D` | Adaptive average pooling |
| 43 | `GlobalAvgPool` | Global average pooling |

#### 4.1.8 Shape Operations (13 ops)

| # | Op | Description | FLOPs |
|---|-----|-------------|-------|
| 44 | `Reshape` | Change shape | 0 |
| 45 | `Transpose` | Transpose | 0 |
| 46 | `Concat` | Concatenate | 0 |
| 47 | `Split` | Split | 0 |
| 48 | `Gather` | Advanced indexing | 0 |
| 49 | `Scatter` | Indexed write | 0 |
| 50 | `Squeeze` | Remove dim=1 | 0 |
| 51 | `Unsqueeze` | Add dim=1 | 0 |
| 52 | `Permute` | Permute dimensions | 0 |
| 53 | `Expand` | Broadcast expansion | 0 |
| 54 | `Slice` | Slice | 0 |
| 55 | `Pad` | Padding | 0 |
| 56 | `Tile` | Repeat | 0 |

```rust
assert!(TensorOp::Reshape.is_zero_flop());
```

#### 4.1.9 Constants (5 ops)

| # | Op | Description |
|---|-----|-------------|
| 57 | `Constant` | Constant tensor |
| 58 | `Zeros` | Zero tensor |
| 59 | `Ones` | Ones tensor |
| 60 | `Arange` | Sequence [0, 1, ..., n-1] |
| 61 | `Full` | Tensor filled with a value |

#### 4.1.10 Recurrent (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 62 | `LSTMCell` | LSTM cell |
| 63 | `GRUCell` | GRU cell |
| 64 | `RNNCell` | Simple RNN cell |

#### 4.1.11 Advanced Mathematics (9 ops)

| # | Op | Description |
|---|-----|-------------|
| 65 | `Einsum` | Einstein notation |
| 66 | `FFT` | Fast Fourier Transform |
| 67 | `IFFT` | Inverse FFT |
| 68 | `SVD` | Singular Value Decomposition |
| 69 | `Eig` | Eigendecomposition |
| 70 | `Solve` | Linear system solver |
| 71 | `TopK` | Top-K values |
| 72 | `Sort` | Sort |
| 73 | `Cumsum` | Cumulative sum |

#### 4.1.12 Quantisation (6 ops)

| # | Op | Description |
|---|-----|-------------|
| 74 | `Quantize` | FP → INT8 |
| 75 | `Dequantize` | INT8 → FP |
| 76 | `QuantizeInt4` | FP → INT4 |
| 77 | `DequantizeInt4` | INT4 → FP |
| 78 | `QuantizeFp8` | FP → FP8 |
| 79 | `DequantizeFp8` | FP8 → FP |

#### 4.1.13 Diffusion / Generative (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 80 | `UNetDownBlock` | U-Net down block |
| 81 | `UNetUpBlock` | U-Net up block |
| 82 | `TimestepEmbedding` | Timestep embedding (Stable Diffusion) |

#### 4.1.14 GNN — Graph Neural Networks (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 83 | `GNNMessagePassing` | GNN message passing |
| 84 | `GNNGlobalPooling` | GNN global pooling |

#### 4.1.15 MoE — Mixture of Experts (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 85 | `MoEDispatch` | Route to experts |
| 86 | `MoECombine` | Combine expert outputs |

#### 4.1.16 Memory and Gradient (11 ops)

| # | Op | Description |
|---|-----|-------------|
| 87 | `Checkpoint` | Gradient checkpointing (memory saving) |
| 88 | `Offload` | CPU offload (for large models) |
| 89 | `GradAccumulate` | Gradient accumulation |
| 90 | `GradMatMul` | MatMul gradient |
| 91 | `GradReLU` | ReLU gradient |
| 92 | `GradSoftmax` | Softmax gradient |
| 93 | `GradLayerNorm` | LayerNorm gradient |
| 94 | `GradAttention` | Attention gradient |
| 95 | `GradConv2D` | Conv2D gradient |
| 96 | `GradLinear` | Linear gradient |
| 97 | `GradGeLU` | GeLU gradient |

#### 4.1.17 Parallelism (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 98 | `ParallelSplit` | Data parallel split |
| 99 | `ParallelAllReduce` | All-reduce across GPUs |
| 100 | `PipelineSend` | Pipeline parallel send |
| 101 | `PipelineReceive` | Pipeline parallel receive |

#### 4.1.18 Fused Operations (6 ops)

| # | Op | Description | Gain |
|---|-----|-------------|------|
| 102 | `FusedMatMulBiasReLU` | MatMul + Bias + ReLU | 1 kernel instead of 3 |
| 103 | `FusedMatMulBias` | MatMul + Bias | 1 kernel instead of 2 |
| 104 | `FusedLinearGeLU` | Linear + GeLU | Bandwidth gain |
| 105 | `FusedAttentionLayerNorm` | Attention + LayerNorm | Memory reduction |
| 106 | `FusedLinearSiLU` | Linear + SiLU | Bandwidth gain |
| 107 | `FusedConvBatchNormReLU` | Conv + BN + ReLU | Fast inference |

### 4.2 Shape Inference

```rust
use lift_core::types::*;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// Shape inference
let a = mk(vec![2, 3, 64], DataType::FP32);
let b = mk(vec![2, 64, 128], DataType::FP32);
let result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).unwrap();
// result[0].shape = [2, 3, 128]

// FLOP computation
let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
println!("FLOPs: {:?}", flops); // Some(49152)

// Memory computation
let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
println!("Memory: {:?} bytes", mem);
```

**Combine with**: `lift-sim` (cost model uses FLOPs), `lift-predict` (roofline prediction).

### 4.3 Useful Predicates

```rust
let op = TensorOp::FlashAttention;

op.is_attention();      // true — attention variant?
op.is_convolution();    // false — convolution?
op.is_normalisation();  // false — normalisation?
op.is_activation();     // false — activation?
op.is_fused();          // false — fused operation?
op.is_gradient();       // false — gradient operation?
op.is_zero_flop();      // false — zero FLOPs (reshape, etc.)?
op.num_inputs();        // (3, 4) — min/max number of inputs
op.flops_formula();     // "2*B*H*(S^2*D + S*D^2)"
```

---

## 5. lift-quantum — Quantum Gates and Noise (50+ gates)

### 5.1 Quantum Gates

#### 5.1.1 Standard 1-Qubit Gates (9 gates)

| # | Gate | IR Name | Type | Description |
|---|-------|--------|------|-------------|
| 1 | `H` | `quantum.h` | Clifford | Hadamard |
| 2 | `X` | `quantum.x` | Pauli | Quantum NOT (bit-flip) |
| 3 | `Y` | `quantum.y` | Pauli | Y rotation by π |
| 4 | `Z` | `quantum.z` | Pauli | Phase-flip |
| 5 | `S` | `quantum.s` | Clifford | Phase π/2 |
| 6 | `Sdg` | `quantum.sdg` | Clifford | S inverse |
| 7 | `T` | `quantum.t` | Non-Clifford | Phase π/4 (expensive for QEC) |
| 8 | `Tdg` | `quantum.tdg` | Non-Clifford | T inverse |
| 9 | `SX` | `quantum.sx` | Clifford | Square root of X |

#### 5.1.2 Parametric 1-Qubit Gates (9 gates)

| # | Gate | Parameters | Description |
|---|------|------------|-------------|
| 10 | `RX` | θ | Rotation around X |
| 11 | `RY` | θ | Rotation around Y |
| 12 | `RZ` | θ | Rotation around Z |
| 13 | `P` | φ | Phase gate |
| 14 | `U1` | λ | U1 unitary gate |
| 15 | `U2` | φ, λ | U2 unitary gate |
| 16 | `U3` | θ, φ, λ | General unitary gate |
| 17 | `Rx90` | — | Fixed RX(π/2) |
| 18 | `Rx180` | — | Fixed RX(π) |

#### 5.1.3 Portes 2-qubits (14 portes)

| # | Porte | Description | Natif pour |
|---|-------|-------------|-----------|
| 19 | `CX` | CNOT | IBM |
| 20 | `CZ` | Controlled-Z | IBM, Rigetti |
| 21 | `CY` | Controlled-Y | — |
| 22 | `SWAP` | Échange de qubits | — |
| 23 | `ISWAP` | iSWAP | Rigetti |
| 24 | `ECR` | Echoed Cross-Resonance | IBM Eagle |
| 25 | `RZX` | ZX rotation | IBM |
| 26 | `XX` | Ising XX | IonQ |
| 27 | `YY` | Ising YY | IonQ |
| 28 | `ZZ` | Ising ZZ | IonQ |
| 29 | `CPhase` | Controlled Phase | Rigetti |
| 30 | `XY` | XY interaction | Rigetti |
| 31 | `CP` | Controlled Phase | — |
| 32 | `MS` | Mølmer–Sørensen | IonQ |

#### 5.1.4 3-Qubit and Multi-Control Gates (4 gates)

| # | Gate | Description |
|---|-------|-------------|
| 33 | `CCX` | Toffoli (CCNOT) |
| 34 | `CSWAP` | Fredkin |
| 35 | `MCX` | Multi-controlled X |
| 36 | `MCZ` | Multi-controlled Z |

#### 5.1.5 Special and Control Gates (10 gates)

| # | Gate | Description |
|---|------|-------------|
| 37 | `GlobalPhase` | Global phase |
| 38 | `Delay` | Delay (decoherence) |
| 39 | `VirtualRZ` | Virtual RZ (no physical cost) |
| 40 | `IfElse` | Classical conditional control |
| 41 | `Measure` | Measure 1 qubit |
| 42 | `MeasureAll` | Measure all qubits |
| 43 | `Reset` | Reset |
| 44 | `Barrier` | Barrier (prevents optimisation) |
| 45 | `Init` | Initialisation |
| 46 | `ParamGate` | Generic parametric gate |

```rust
use lift_quantum::gates::QuantumGate;

let gate = QuantumGate::H;
println!("Name: {}", gate.op_name());       // "quantum.h"
println!("Qubits: {}", gate.num_qubits());  // 1
println!("Clifford: {}", gate.is_clifford()); // true
println!("Parametric: {}", gate.is_parametric()); // false
println!("Self-inverse: {}", gate.is_self_inverse()); // true

// Look up a gate by its IR name
let gate = QuantumGate::from_name("quantum.cx"); // Some(CX)
```

### 5.2 Hardware Providers — Native Gate Sets

```rust
use lift_quantum::gates::{QuantumGate, Provider};

// Native gates per provider
let ibm_basis = QuantumGate::native_basis(Provider::IbmEagle);
let rigetti_basis = QuantumGate::native_basis(Provider::Rigetti);
let ionq_basis = QuantumGate::native_basis(Provider::IonQ);
let quant_basis = QuantumGate::native_basis(Provider::Quantinuum);
```

| Provider | Native Gates |
|----------|---------------|
| `IbmEagle` | CX, RZ, SX, X |
| `IbmKyoto` | ECR, RZ, SX, X |
| `Rigetti` | CZ, RX, RZ |
| `IonQ` | GPI, GPI2, MS |
| `Quantinuum` | RZ, RX, ZZ |
| `Simulator` | All gates |

**Combine with**: `lift-opt::LayoutMapping` (transpilation to target hardware).

### 5.3 Device Topology — Hardware Topology

```rust
use lift_quantum::topology::DeviceTopology;

// Predefined topologies
let linear = DeviceTopology::linear(10);         // Linear chain
let grid = DeviceTopology::grid(3, 3);           // 3x3 grid
let hex = DeviceTopology::heavy_hex(27);         // Heavy-hex IBM
let ion = DeviceTopology::all_to_all(32);        // All-to-all (trapped ions)
let tree = DeviceTopology::tree(15);             // Binary tree

// Custom topology
let custom = DeviceTopology::custom("my_chip",
    &[(0,1), (1,2), (2,3), (0,3)], 0.99);

// Querying
linear.are_connected(0, 1);           // true
linear.shortest_path(0, 4);           // Some([0, 1, 2, 3, 4])
linear.swap_distance(0, 4);           // Some(3)
linear.avg_connectivity();             // average connectivity
linear.diameter();                     // graph diameter
grid.neighbors(4);                     // neighbours of qubit 4
```

**Combine with**: `lift-opt::LayoutMapping`, `lift-opt::NoiseAwareSchedule`.

### 5.4 Noise Models

```rust
use lift_quantum::noise::{NoiseModel, GateNoise, CircuitNoise};

// Noise models
let ideal = NoiseModel::Ideal;
let depol = NoiseModel::Depolarizing { p: 0.01 };
let bitflip = NoiseModel::BitFlip { p: 0.001 };
let phaseflip = NoiseModel::PhaseFlip { p: 0.001 };

// Model fidelity
let fidelity = depol.fidelity(); // 0.99

// Per-gate noise
let gate_noise = GateNoise::with_depolarizing(0.999, 0.02);

// Full circuit analysis
let mut cn = CircuitNoise::new();
// ... noise accumulation
println!("Total fidelity: {}", cn.total_fidelity);
println!("2-qubit gates: {}", cn.two_qubit_count);
```

### 5.5 Kraus Channels — Quantum Noise Channels

```rust
use lift_quantum::kraus::{ComplexMatrix, KrausChannel};

// Predefined noise channels
let depol = KrausChannel::depolarizing(0.01, 1);     // 1-qubit depolarising
let amp = KrausChannel::amplitude_damping(0.02);      // Amplitude damping
let phase = KrausChannel::phase_damping(0.01);         // Phase damping

// Channel fidelity
let fidelity = depol.average_gate_fidelity();
println!("Fidelity: {:.6}", fidelity);

// Complex matrices
let mut m = ComplexMatrix::identity(2);
let dagger = m.dagger();    // Conjugate transpose
let product = m.mul(&dagger).unwrap();
let trace = m.trace().unwrap();
```

**Combine with**: `lift-sim::QuantumCostModel` (circuit fidelity estimation), `lift-opt::NoiseAwareSchedule`.

### 5.6 QEC — Quantum Error Correction

```rust
use lift_quantum::qec::{QecCode, QecAnalysis};

// Available QEC codes
let surface = QecCode::SurfaceCode { distance: 5 };   // 25 physical qubits/logical
let steane = QecCode::SteaneCode;                       // 7 physical qubits/logical
let shor = QecCode::ShorCode;                            // 9 physical qubits/logical
let rep = QecCode::RepetitionCode { distance: 7 };     // 7 physical qubits
let ldpc = QecCode::LdpcCode { n: 100, k: 10 };       // LDPC code

// Code properties
println!("Physical/logical: {}", surface.physical_per_logical()); // 25
println!("Distance: {}", surface.code_distance());                 // 5
println!("Syndrome depth: {}", surface.syndrome_circuit_depth()); // 5

// Full QEC analysis
let analysis = QecAnalysis::analyse(
    10,     // logical qubits
    100,    // circuit depth
    QecCode::SurfaceCode { distance: 5 },
    0.001,  // physical error rate
);
println!("Physical qubits: {}", analysis.physical_qubits);
println!("Logical error rate: {:.2e}", analysis.logical_error_rate);
println!("Overhead: {}", analysis.overhead_qubits);
```

**Combine with**: `lift-sim::QuantumCostModel`, `lift-predict` (fidelity budget).

---

## 6. lift-hybrid — Classical-Quantum Hybrid Computation

### 6.1 Hybrid Operations (21 ops)

#### 6.1.1 Encoding/Decoding (2 ops)

| # | Op | IR Name | Description |
|---|-----|---------|-------------|
| 1 | `Encode` | `hybrid.encode` | Encode classical data → qubits |
| 2 | `Decode` | `hybrid.decode` | Decode quantum measurements → classical |

#### 6.1.2 Gradient Methods (6 ops)

| # | Op | IR Name | Evaluations | Exact? |
|---|-----|---------|-------------|--------|
| 3 | `ParameterShift` | `hybrid.parameter_shift` | 2N | Yes |
| 4 | `FiniteDifference` | `hybrid.finite_difference` | N+1 | No |
| 5 | `SPSA` | `hybrid.spsa` | 2 | No |
| 6 | `AdjointDifferentiation` | `hybrid.adjoint_diff` | 1 | Yes |
| 7 | `StochasticParameterShift` | `hybrid.stochastic_param_shift` | 2 | No |
| 8 | `JointGradient` | `hybrid.joint_gradient` | Combined | — |

```rust
use lift_hybrid::gradient::GradientMethod;

let method = GradientMethod::ParameterShift;
let evals = method.circuit_evaluations(100); // 200 evaluations for 100 params
assert!(method.is_exact()); // true
```

#### 6.1.3 Processing (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 9 | `ClassicalPreprocess` | Classical preprocessing |
| 10 | `QuantumPostprocess` | Quantum postprocessing |
| 11 | `HybridForward` | Hybrid forward pass |
| 12 | `HybridBackward` | Hybrid backward pass |

#### 6.1.4 Variational Algorithms (4 ops)

| # | Op | Description | Usage |
|---|-----|-------------|-------|
| 13 | `VqcLayer` | Variational circuit layer | Quantum classification |
| 14 | `VqeAnsatz` | VQE ansatz | Quantum chemistry |
| 15 | `QaoaLayer` | QAOA layer | Combinatorial optimisation |
| 16 | `QuantumKernel` | Quantum kernel | Quantum machine learning |

#### 6.1.5 Data Transfer (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 17 | `GpuToQpu` | GPU → QPU transfer |
| 18 | `QpuToGpu` | QPU → GPU transfer |

#### 6.1.6 Co-Execution and Measurement (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 19 | `CoExecute` | Simultaneous classical+quantum execution |
| 20 | `MeasureExpectation` | Observable expectation value |
| 21 | `MeasureSamples` | Measurement sampling |

```rust
use lift_hybrid::ops::HybridOp;

let op = HybridOp::VqcLayer;
assert!(op.is_variational());
assert!(!op.is_gradient());
```

### 6.2 Encoding Strategies

```rust
use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

let strategies = [
    EncodingStrategy::AngleEncoding,       // 1 qubit/feature, depth 1
    EncodingStrategy::AmplitudeEncoding,    // log2(n) qubits, depth n
    EncodingStrategy::BasisEncoding,        // 1 qubit/feature, depth 1
    EncodingStrategy::IQPEncoding,          // 1 qubit/feature, depth 2n
    EncodingStrategy::HamiltonianEncoding,  // 1 qubit/feature, depth n
    EncodingStrategy::KernelEncoding,       // 1 qubit/feature, depth 3n
];

// Encoding configuration
let config = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 256);
println!("Qubits required: {}", config.num_qubits); // 8 = log2(256)
println!("Classical dimension: {}", config.classical_dim); // 256
```

| Strategy | Qubits | Depth | Best for |
|----------|--------|-------|----------|
| Angle | n | 1 | Few features |
| Amplitude | log₂(n) | n | Many features |
| Basis | n | 1 | Binary data |
| IQP | n | 2n | Quantum advantage |
| Hamiltonian | n | n | Physical simulation |
| Kernel | n | 3n | Quantum ML |

### 6.3 Gradient Configuration — Joint Gradient Setup

```rust
use lift_hybrid::gradient::{GradientMethod, JointGradientConfig};

let config = JointGradientConfig {
    classical_method: GradientMethod::Backprop,
    quantum_method: GradientMethod::ParameterShift,
    num_classical_params: 1000,
    num_quantum_params: 50,
};
println!("Total evaluations: {}", config.total_evaluations());
// 1 (backprop) + 100 (2*50 parameter shift) = 101
```

### 6.4 Auxiliary Types

```rust
use lift_hybrid::ops::{AnsatzType, SyncPolicy, FeatureMap};

// Ansatz types for VQC
let ansatz = AnsatzType::HardwareEfficient; // HardwareEfficient, StronglyEntangling, TwoLocal, UCCSD, Custom

// Synchronisation policy
let sync = SyncPolicy::Blocking; // Blocking, Asynchronous, Pipeline

// Feature maps for quantum kernels
let fm = FeatureMap::ZZFeatureMap; // ZZFeatureMap, PauliFeatureMap, AngleEncoding, AmplitudeEncoding
```

---

## 7. lift-opt — Optimisation Passes (11 passes)

### 7.1 Classical Passes (5 passes)

#### 7.1.1 Canonicalize — Canonical Form

```rust
use lift_opt::Canonicalize;
use lift_core::pass::Pass;

let pass = Canonicalize;
// Reorders operations into canonical form
// Normalises IR patterns to facilitate subsequent optimisations
```

**Usage**: Always run first in the pipeline.

#### 7.1.2 ConstantFolding — Constant Folding

```rust
use lift_opt::ConstantFolding;

let pass = ConstantFolding;
// Evaluates operations whose operands are all compile-time constants
// Example: add(const(2), const(3)) → const(5)
```

#### 7.1.3 DeadCodeElimination — Dead Code Elimination

```rust
use lift_opt::DeadCodeElimination;

let pass = DeadCodeElimination;
// Removes operations whose results are never used
// Respects operations with side effects (measurements, etc.)
```

#### 7.1.4 TensorFusion — Tensor Fusion

```rust
use lift_opt::TensorFusion;

let pass = TensorFusion;
// Fuses consecutive operations into fused operations
// Example: MatMul + Bias + ReLU → FusedMatMulBiasReLU
// Reduces memory accesses and kernel launches
```

**Combine with**: Run after `Canonicalize` and `ConstantFolding`.

#### 7.1.5 CommonSubexprElimination — Common Subexpression Elimination

```rust
use lift_opt::CommonSubexprElimination;

let pass = CommonSubexprElimination;
// Detects identical operations (same op, same operands)
// Replaces duplicates with references to the first occurrence
// Excludes operations with side effects
```

### 7.2 Quantum Passes (3 passes)

#### 7.2.1 GateCancellation — Gate Cancellation

```rust
use lift_opt::GateCancellation;

let pass = GateCancellation;
// Removes gate pairs that cancel out
// Example: H H → identity, X X → identity
// Respects qubit linearity invariants
```

#### 7.2.2 RotationMerge — Rotation Merging

```rust
use lift_opt::RotationMerge;

let pass = RotationMerge;
// Merges consecutive rotations on the same axis
// Example: RZ(0.3) RZ(0.5) → RZ(0.8)
// Removes identity rotations (angle ≈ 0)
```

#### 7.2.3 NoiseAwareSchedule — Noise-Aware Scheduling

```rust
use lift_opt::NoiseAwareSchedule;

let pass = NoiseAwareSchedule;
// Reorders quantum gates to minimise decoherence
// Prioritises fast gates (1-qubit) before slow ones (2-qubit)
// Respects SSA dependencies
```

**Combine with**: `lift-quantum::topology::DeviceTopology` for the target topology.

### 7.3 Advanced AI Passes (3 passes)

#### 7.3.1 FlashAttentionPass — FlashAttention Replacement

```rust
use lift_opt::FlashAttentionPass;

let pass = FlashAttentionPass::default(); // threshold = 512
let pass_custom = FlashAttentionPass { seq_len_threshold: 1024 };
// Replaces tensor.attention with tensor.flash_attention
// when sequence length exceeds the threshold
// Reduces memory complexity from O(N²) to O(N)
```

#### 7.3.2 QuantisationPass — Quantisation Annotation

```rust
use lift_opt::QuantisationPass;
use lift_opt::quantisation_pass::{QuantTarget, QuantMode};

let pass = QuantisationPass::default(); // INT8, Dynamic
let pass_custom = QuantisationPass {
    target_dtype: QuantTarget::Fp8E4M3,
    mode: QuantMode::Static,
};
// Annotates heavy operations (MatMul, Conv, Linear, Attention)
// with quantisation metadata
// Inserts Quantize/Dequantize pairs around annotated ops
```

| Target | Size | Usage |
|--------|------|-------|
| `Int8` | 1 byte | Standard inference |
| `Int4` | 0.5 byte | Compressed LLMs (GPTQ, AWQ) |
| `Fp8E4M3` | 1 byte | H100 training |
| `Fp8E5M2` | 1 byte | H100 inference |

#### 7.3.3 LayoutMapping — Qubit Mapping

```rust
use lift_opt::LayoutMapping;

let pass = LayoutMapping;
// Inserts SWAP gates to map logical qubits to physical qubits
// Based on the target device topology
// Marks operations requiring swaps via attributes
```

**Combine with**: `lift-quantum::topology::DeviceTopology`.

### 7.4 Recommended Optimisation Pipeline

```rust
use lift_core::PassManager;

let mut pm = PassManager::new();

// Phase 1: Cleanup
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::ConstantFolding));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.add_pass(Box::new(lift_opt::CommonSubexprElimination));

// Phase 2: Fusion (AI)
pm.add_pass(Box::new(lift_opt::TensorFusion));
pm.add_pass(Box::new(lift_opt::FlashAttentionPass::default()));
pm.add_pass(Box::new(lift_opt::QuantisationPass::default()));

// Phase 3: Quantum
pm.add_pass(Box::new(lift_opt::GateCancellation));
pm.add_pass(Box::new(lift_opt::RotationMerge));
pm.add_pass(Box::new(lift_opt::NoiseAwareSchedule));
pm.add_pass(Box::new(lift_opt::LayoutMapping));

// Phase 4: Final cleanup
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

let results = pm.run_all(&mut ctx);
```

---

## 8. lift-sim — Simulation and Cost Analysis

### 8.1 CostModel — Classical Cost Model

```rust
use lift_sim::cost::CostModel;

// Predefined GPU profiles
let a100 = CostModel::a100();  // 312 TFLOPS, 2039 GB/s
let h100 = CostModel::h100();  // 989 TFLOPS, 3350 GB/s

// Time estimation
let flops = 2 * 1024 * 1024 * 1024_u64;
let bytes = 4 * 1024 * 1024_u64;

let compute_ms = a100.compute_time_ms(flops);      // Compute time
let memory_ms = a100.memory_time_ms(bytes);         // Memory time
let roofline_ms = a100.roofline_time_ms(flops, bytes); // Roofline model

// Analysis
let ai = a100.arithmetic_intensity(flops, bytes);   // FLOPs/byte
let bound = a100.is_compute_bound(flops, bytes);    // true = compute-bound
let fits = a100.fits_in_memory(bytes);               // Fits in memory?
let gpus = a100.num_gpus_needed(bytes);              // GPUs needed
```

### 8.2 QuantumCostModel — Quantum Cost Model

```rust
use lift_sim::cost::QuantumCostModel;

// Quantum processor profiles
let sc = QuantumCostModel::superconducting_default(); // IBM-like: 127 qubits
let ion = QuantumCostModel::trapped_ion_default();     // IonQ-like: 32 qubits
let atom = QuantumCostModel::neutral_atom_default();   // Atom-like: 256 qubits

// Circuit fidelity estimation
let fidelity = sc.circuit_fidelity(50, 20); // 50 1Q gates, 20 2Q gates
println!("Fidelity: {:.6}", fidelity);

// Circuit time
let time_us = sc.circuit_time_us(50, 20, 5, 10); // 50 1Q, 20 2Q, 5 measurements, 10 depth
println!("Time: {:.2} µs", time_us);

// Decoherence fidelity
let decoherence = sc.decoherence_fidelity(time_us);
println!("Decoherence fidelity: {:.6}", decoherence);
```

| Parameter | Superconducting | Trapped Ions | Neutral Atoms |
|-----------|----------------|--------------|---------------|
| 1Q time | 0.02 µs | 10 µs | 0.5 µs |
| 2Q time | 0.3 µs | 200 µs | 1.0 µs |
| 1Q fidelity | 99.9% | 99.99% | 99.9% |
| 2Q fidelity | 99% | 99.9% | 99.5% |
| T1 | 100 µs | 1 s | 5 ms |
| Qubits | 127 | 32 | 256 |

### 8.3 Budget — Resource Constraints

```rust
use lift_sim::cost::Budget;

let budget = Budget {
    max_flops: Some(1_000_000_000_000), // 1 TFLOP max
    max_memory_bytes: Some(80_000_000_000), // 80 GB
    max_time_ms: Some(100.0),           // 100 ms
    min_fidelity: Some(0.99),           // 99% min fidelity
    max_circuit_depth: Some(1000),      // 1000 layers max
};

budget.check_flops(500_000_000_000).unwrap();   // OK
budget.check_memory(40_000_000_000).unwrap();   // OK
budget.check_fidelity(0.995).unwrap();           // OK
```

### 8.4 EnergyModel — Energy and Carbon Estimation

```rust
use lift_sim::cost::EnergyModel;

let model = EnergyModel::a100();

// Energy for 1 second of computation on 4 GPUs
let joules = model.energy_joules(1000.0, 4);     // Joules
let kwh = model.energy_kwh(1000.0, 4);           // kWh
let carbon = model.carbon_grams(1000.0, 4);       // grams CO₂

println!("Energy: {:.2} J", joules);
println!("Carbon: {:.4} g CO₂", carbon);

// Quantum energy (cryogenic refrigeration)
let q_joules = model.quantum_energy_joules(100.0, 127); // 100 µs, 127 qubits
```

### 8.5 ReactiveBudget — Dynamic Budget

```rust
use lift_sim::cost::{Budget, ReactiveBudget};

let budget = Budget {
    max_flops: Some(1_000_000),
    max_memory_bytes: Some(1_000_000),
    max_time_ms: Some(50.0),
    min_fidelity: Some(0.9),
    max_circuit_depth: None,
};
let mut rb = ReactiveBudget::new(budget);

// Consume resources incrementally
rb.consume(100_000, 50_000, 5.0, 0.99); // flops, mem, time, fidelity
rb.consume(200_000, 80_000, 10.0, 0.98);

// Check remaining budget
rb.check_remaining().unwrap(); // OK if within limits

// Utilisation report
let util = rb.utilisation();
println!("FLOPs used: {:.1}%", util.flop_ratio.unwrap() * 100.0);
println!("Time used: {:.1}%", util.time_ratio.unwrap() * 100.0);

// Remaining budget
println!("Remaining FLOPs: {:?}", rb.remaining_flops());
println!("Remaining time: {:?} ms", rb.remaining_time_ms());
```

**Combine with**: `lift-opt` (stop optimisation if budget exhausted), `lift-predict` (verify prediction respects budget).

### 8.6 Module Analysis

```rust
use lift_sim::{analyze_module, analyze_quantum_ops};

let ctx = load_and_parse("model.lif").unwrap();

// Classical analysis
let report = analyze_module(&ctx);
println!("Total ops: {}", report.num_ops);
println!("Tensor ops: {}", report.num_tensor_ops);
println!("Quantum ops: {}", report.num_quantum_ops);
println!("Hybrid ops: {}", report.num_hybrid_ops);
println!("Total FLOPs: {}", report.total_flops);
println!("Total memory: {} bytes", report.total_memory_bytes);
println!("Peak memory: {} bytes", report.peak_memory_bytes);

// Quantum analysis
let quantum = analyze_quantum_ops(&ctx);
println!("Qubits: {}", quantum.num_qubits_used);
println!("Gates: {}", quantum.gate_count);
println!("1Q gates: {}", quantum.one_qubit_gates);
println!("2Q gates: {}", quantum.two_qubit_gates);
println!("Measurements: {}", quantum.measurements);
println!("Estimated fidelity: {:.6}", quantum.estimated_fidelity);
```

---

## 9. lift-predict — Performance Prediction

```rust
use lift_predict::predict_performance;
use lift_sim::{analyze_module, cost::CostModel};

let report = analyze_module(&ctx);
let cost_model = CostModel::h100();
let prediction = predict_performance(&report, &cost_model);

println!("Compute time: {:.4} ms", prediction.compute_time_ms);
println!("Memory time: {:.4} ms", prediction.memory_time_ms);
println!("Predicted time: {:.4} ms", prediction.predicted_time_ms);
println!("Arithmetic intensity: {:.2} FLOP/byte", prediction.arithmetic_intensity);
println!("Bottleneck: {}", prediction.bottleneck); // "compute" or "memory"
```

**Combine with**: `lift-sim` (provides the analysis report and cost model).

---

## 10. lift-import — Model Import

### 10.1 ONNX Import

```rust
use lift_import::OnnxImporter;

let importer = OnnxImporter::new();
let ctx = importer.import("model.onnx").expect("ONNX import failed");
```

### 10.2 PyTorch FX Import

```rust
use lift_import::PyTorchFxImporter;

let importer = PyTorchFxImporter::new();
let ctx = importer.import("model_fx.json").expect("FX import failed");
```

### 10.3 OpenQASM 3.0 Import

```rust
use lift_import::OpenQasm3Importer;

let importer = OpenQasm3Importer::new();
let ctx = importer.import("circuit.qasm").expect("QASM import failed");
```

**Combine with**: `lift-core::verifier` (verify imported IR), then `lift-opt` (optimise).

---

## 11. lift-export — Backend Export

### 11.1 Export LLVM IR

```rust
use lift_export::LlvmExporter;

let exporter = LlvmExporter::new();
let llvm_ir = exporter.export(&ctx).expect("LLVM export failed");
std::fs::write("output.ll", &llvm_ir).unwrap();
```

Produces LLVM IR compilable with `clang` or `llc`.

### 11.2 Export OpenQASM 3.0

```rust
use lift_export::QasmExporter;

let exporter = QasmExporter::new();
let qasm = exporter.export(&ctx).expect("QASM export failed");
std::fs::write("output.qasm", &qasm).unwrap();
```

Produces OpenQASM 3.0 executable on IBM Quantum, Rigetti, etc.

**Combine with**: `lift-opt` (optimise before export), `lift-quantum::Provider` (transpile to native gate set).

---

## 12. lift-config — Configuration (.lith)

### 12.1 .lith File Format

```ini
[target]
backend = "cuda"
device = "A100"
precision = "fp16"

[budget]
max_flops = 1000000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0
min_fidelity = 0.99

[optimisation]
level = O2
max_iterations = 10

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = "heavy_hex"
num_qubits = 127
shots = 4096
```

### 12.2 Programmatic Loading

```rust
use lift_config::{ConfigParser, LithConfig};

// From a file
let source = std::fs::read_to_string("config.lith").unwrap();
let config = ConfigParser::new().parse(&source).unwrap();

// Default configuration
let default = LithConfig::default();
// Backend: llvm, Level: O2, Passes: canonicalize, constant-folding, dce, tensor-fusion

// With quantum
let hybrid = LithConfig::default().with_quantum("heavy_hex", 127);
```

### 12.3 Optimisation Levels

| Level | Passes | Usage |
|-------|--------|-------|
| `O0` | None | Debug, verification |
| `O1` | Canonicalize, DCE | Fast compilation |
| `O2` | + ConstantFolding, TensorFusion | **Default** — good trade-off |
| `O3` | + FlashAttention, Quantisation, CSE | Maximum performance |

---

## 13. lift-cli — Command-Line Interface

### 13.1 Available Commands

#### 13.1.1 `lift verify` — Verify a .lif file

```bash
lift verify model.lif
lift verify --verbose model.lif
```

Checks SSA invariants, qubit linearity, and typing.

#### 13.1.2 `lift analyse` — Analyse a program

```bash
lift analyse model.lif
lift analyse model.lif --format json
```

Produces a report: op count, FLOPs, memory, quantum analysis.

#### 13.1.3 `lift print` — Display the IR

```bash
lift print model.lif
```

Affiche l'IR en format lisible.

#### 13.1.4 `lift optimise` — Optimiser

```bash
lift optimise model.lif
lift optimise model.lif --config config.lith --output optimised.lif
```

Applique les passes d'optimisation configurées.

#### 13.1.5 `lift predict` — Prédire la performance

```bash
lift predict model.lif --device a100
lift predict model.lif --device h100
```

Prédit le temps d'exécution avec le modèle roofline.

#### 13.1.6 `lift export` — Exporter

```bash
lift export model.lif --backend llvm --output model.ll
lift export quantum.lif --backend qasm --output circuit.qasm
```

Exporte vers LLVM IR ou OpenQASM 3.0.

---

## 14. Combinaisons et pipelines complets

### 14.1 Pipeline IA complet (Transformer)

```rust
// 1. Importer un modèle ONNX
let ctx = OnnxImporter::new().import("bert.onnx")?;

// 2. Vérifier
verifier::verify(&ctx)?;

// 3. Analyser
let report = analyze_module(&ctx);

// 4. Optimiser
let mut pm = PassManager::new();
pm.add_pass(Box::new(Canonicalize));
pm.add_pass(Box::new(ConstantFolding));
pm.add_pass(Box::new(DeadCodeElimination));
pm.add_pass(Box::new(CommonSubexprElimination));
pm.add_pass(Box::new(TensorFusion));
pm.add_pass(Box::new(FlashAttentionPass { seq_len_threshold: 512 }));
pm.add_pass(Box::new(QuantisationPass {
    target_dtype: QuantTarget::Int8,
    mode: QuantMode::Dynamic,
}));
pm.add_pass(Box::new(DeadCodeElimination));
pm.run_all(&mut ctx);

// 5. Prédire la performance
let h100 = CostModel::h100();
let pred = predict_performance(&analyze_module(&ctx), &h100);

// 6. Exporter vers LLVM
let llvm = LlvmExporter::new().export(&ctx)?;
std::fs::write("bert_optimised.ll", llvm)?;
```

### 14.2 Pipeline quantique complet (Bell State)

```rust
// 1. Parser le circuit
let ctx = load_lif_file("quantum_bell.lif")?;

// 2. Analyser le bruit
let quantum = analyze_quantum_ops(&ctx);
let sc = QuantumCostModel::superconducting_default();
let fidelity = sc.circuit_fidelity(
    quantum.one_qubit_gates, quantum.two_qubit_gates
);

// 3. QEC si nécessaire
if fidelity < 0.99 {
    let analysis = QecAnalysis::analyse(2, 5,
        QecCode::SurfaceCode { distance: 3 }, 0.001);
    println!("Qubits physiques nécessaires: {}", analysis.physical_qubits);
}

// 4. Optimiser
let mut pm = PassManager::new();
pm.add_pass(Box::new(GateCancellation));
pm.add_pass(Box::new(RotationMerge));
pm.add_pass(Box::new(NoiseAwareSchedule));
pm.add_pass(Box::new(LayoutMapping));
pm.run_all(&mut ctx);

// 5. Exporter vers QASM
let qasm = QasmExporter::new().export(&ctx)?;
std::fs::write("bell_optimised.qasm", qasm)?;
```

### 14.3 Pipeline hybride complet (VQE)

```rust
// 1. Configurer l'encodage
let encoding = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);

// 2. Configurer le gradient
let grad_config = JointGradientConfig {
    classical_method: GradientMethod::Backprop,
    quantum_method: GradientMethod::ParameterShift,
    num_classical_params: 100,
    num_quantum_params: 20,
};

// 3. Budget réactif pour contrôler les ressources
let budget = Budget {
    max_flops: Some(1_000_000_000),
    max_memory_bytes: Some(8_000_000_000),
    max_time_ms: Some(60_000.0),
    min_fidelity: Some(0.95),
    max_circuit_depth: Some(500),
};
let mut rb = ReactiveBudget::new(budget);

// 4. Boucle d'optimisation VQE
for iteration in 0..100 {
    // Exécuter le circuit quantique
    rb.consume(10_000, 1_000, 0.5, 0.999);
    
    if rb.check_remaining().is_err() {
        println!("Budget épuisé à l'itération {}", iteration);
        break;
    }
    
    let util = rb.utilisation();
    println!("Itération {}: FLOP {:.1}%, Temps {:.1}%",
        iteration,
        util.flop_ratio.unwrap() * 100.0,
        util.time_ratio.unwrap() * 100.0
    );
}

// 5. Estimer l'empreinte carbone
let energy = EnergyModel::a100();
let carbon = energy.carbon_grams(rb.elapsed_ms, 1);
println!("Empreinte carbone: {:.4} g CO₂", carbon);
```

### 14.4 Pipeline CLI complet

```bash
# Vérifier, analyser, optimiser, prédire et exporter en une séquence
lift verify model.lif
lift analyse model.lif --format json > analysis.json
lift optimise model.lif --config production.lith --output optimised.lif
lift predict optimised.lif --device h100
lift export optimised.lif --backend llvm --output model.ll
```

---

## 15. Exemples concrets

### 15.1 MLP (Perceptron multi-couches)

Fichier `tensor_mlp.lif` :

```
#dialect tensor

module @mlp {
    func @forward(%x: tensor<1x784xf32>, %w1: tensor<784x256xf32>,
                  %b1: tensor<256xf32>, %w2: tensor<256x10xf32>,
                  %b2: tensor<10xf32>) -> tensor<1x10xf32> {
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

### 15.2 Self-Attention (Transformer)

Fichier `attention.lif` :

```
#dialect tensor

module @transformer {
    func @self_attention(%q: tensor<1x128x64xf32>, %k: tensor<1x128x64xf32>,
                         %v: tensor<1x128x64xf32>, %norm_w: tensor<64xf32>)
                         -> tensor<1x128x64xf32> {
        %attn = "tensor.attention"(%q, %k, %v) : (...) -> tensor<1x128x64xf32>
        %normed = "tensor.layernorm"(%attn, %norm_w) : (...) -> tensor<1x128x64xf32>
        return %normed
    }
}
```

### 15.3 État de Bell (Quantique)

Fichier `quantum_bell.lif` :

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

### 15.4 Configuration de production

Fichier `production.lith` :

```ini
[target]
backend = "cuda"
device = "H100"
precision = "fp16"

[budget]
max_flops = 1000000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0

[optimisation]
level = O3
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
```

---

## Résumé des combinaisons par tâche

| Tâche | Crates à combiner |
|-------|-------------------|
| **Entraîner un LLM** | lift-tensor + lift-opt (TensorFusion, FlashAttention) + lift-sim (CostModel) + lift-export (LLVM) |
| **Inférence quantisée** | lift-tensor + lift-opt (QuantisationPass) + lift-predict + lift-export (LLVM) |
| **Circuit quantique** | lift-quantum + lift-opt (GateCancellation, RotationMerge, LayoutMapping) + lift-export (QASM) |
| **VQE / QAOA** | lift-hybrid + lift-quantum + lift-opt (NoiseAwareSchedule) + lift-sim (QuantumCostModel) |
| **Quantum ML** | lift-hybrid (QuantumKernel, encoding) + lift-tensor + lift-quantum |
| **Analyse de coût** | lift-sim (CostModel, EnergyModel) + lift-predict |
| **QEC planning** | lift-quantum (qec, topology) + lift-sim (QuantumCostModel) |
| **Import/Optimise/Export** | lift-import + lift-opt + lift-export |
| **Stable Diffusion** | lift-tensor (UNet ops) + lift-opt (TensorFusion) + lift-export |
| **GNN** | lift-tensor (GNNMessagePassing, GNNGlobalPooling) + lift-opt + lift-export |
