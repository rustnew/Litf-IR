# LIFT — Language for Intelligent Frameworks and Technologies

> **Unified intermediate representation for AI and quantum computing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.2.1-green.svg)](Cargo.toml)

LIFT is a modular compiler framework that provides a single SSA-based intermediate representation spanning **tensor operations** (AI/ML), **quantum gates**, and **classical-quantum hybrid computation**. It enables a unified pipeline: import → verify → optimise → simulate → predict → export.

## Key Features

- **107 tensor operations** — arithmetic, attention (Flash, Paged, GQA), convolutions, normalisation, quantisation, MoE, GNN, diffusion, and more
- **46+ quantum gates** — Pauli, Clifford, parametric, multi-qubit; noise models, Kraus channels, QEC codes
- **21 hybrid operations** — encoding strategies, gradient methods (parameter shift, adjoint), variational algorithms (VQC, VQE, QAOA)
- **11 optimisation passes** — canonicalise, constant folding, DCE, CSE, tensor fusion, FlashAttention replacement, quantisation annotation, gate cancellation, rotation merging, noise-aware scheduling, qubit layout mapping
- **Cost modelling** — roofline analysis, GPU/QPU profiles (A100, H100, IBM, IonQ, etc.), energy/carbon estimation
- **Performance prediction** — compute vs memory bottleneck identification
- **Import/Export** — ONNX, PyTorch FX, OpenQASM 3.0, LLVM IR

## Architecture

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

### Crates

| Crate | Description |
|-------|-------------|
| **lift-core** | SSA IR, type system, verifier, printer, pass manager, dialect registry |
| **lift-ast** | Lexer, parser, IR builder for `.lif` source files |
| **lift-tensor** | 107 tensor operations with shape inference and FLOP counting |
| **lift-quantum** | 46+ quantum gates, hardware providers, device topology, noise models, Kraus channels, QEC |
| **lift-hybrid** | 21 hybrid ops — encoding, gradient methods, variational algorithms, co-execution |
| **lift-opt** | 11 optimisation passes (classical, quantum, and AI-specific) |
| **lift-sim** | Classical/quantum cost models, energy estimation, reactive budgets, module analysis |
| **lift-predict** | Roofline-based performance prediction |
| **lift-import** | ONNX, PyTorch FX, OpenQASM 3.0 importers |
| **lift-export** | LLVM IR and OpenQASM 3.0 exporters |
| **lift-config** | `.lith` configuration file parser |
| **lift-cli** | Command-line interface (`verify`, `analyse`, `optimise`, `predict`, `export`, `print`) |

## Quick Start

### Prerequisites

- **Rust 1.80+** — install via [rustup](https://rustup.rs/)

### Build

```bash
git clone https://github.com/rustnew/Lift.git
cd Lift
cargo build --release
```

### Run

```bash
# Verify a .lif file
cargo run --release -p lift-cli -- verify examples/tensor_mlp.lif

# Analyse
cargo run --release -p lift-cli -- analyse examples/tensor_mlp.lif

# Optimise
cargo run --release -p lift-cli -- optimise examples/tensor_mlp.lif --output optimised.lif

# Predict performance
cargo run --release -p lift-cli -- predict examples/tensor_mlp.lif --device h100

# Export to LLVM IR
cargo run --release -p lift-cli -- export examples/tensor_mlp.lif --backend llvm --output model.ll
```

### Use as a Library

```toml
[dependencies]
lift-core    = "0.2.1"
lift-tensor  = "0.2.1"
lift-quantum = "0.2.1"
lift-opt     = "0.2.1"
```

```rust
use lift_ast::{Lexer, Parser, IrBuilder};
use lift_core::{Context, verifier, pass::PassManager};

// Parse a .lif file
let source = std::fs::read_to_string("model.lif").unwrap();
let tokens = Lexer::new(&source).tokenize().to_vec();
let program = Parser::new(tokens).parse().unwrap();

let mut ctx = Context::new();
IrBuilder::new().build_program(&mut ctx, &program).unwrap();

// Verify
verifier::verify(&ctx).unwrap();

// Optimise
let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::ConstantFolding));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.add_pass(Box::new(lift_opt::TensorFusion));
pm.run_all(&mut ctx);
```

## File Formats

| Extension | Description |
|-----------|-------------|
| `.lif` | LIFT IR source code |
| `.lith` | Compilation configuration |

## Examples

See the [`examples/`](examples/) directory:

- **`tensor_mlp.lif`** — Multi-layer perceptron
- **`attention.lif`** — Transformer self-attention
- **`quantum_bell.lif`** — Bell state preparation

## Documentation

For a comprehensive guide covering every feature, operation, and usage pattern, see **[LIFT_Guide.md](LIFT_Guide.md)**.

## License

[MIT](LICENSE)
