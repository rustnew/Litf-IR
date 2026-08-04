# Contributing to LIFT

Thank you for your interest in contributing to **LIFT — Language for Intelligent
Frameworks and Technologies**. This project is a unified SSA-based intermediate
representation for AI, quantum, and hybrid computation.

## Code of Conduct

Be respectful and constructive. This project is open to everyone.

## Getting Started

```bash
# Clone and build
git clone <your-fork-url>
cd lift
cargo build --workspace

# Run the full test suite (515+ tests)
cargo test --workspace
```

## Repository Layout

| Path | Contents |
|------|----------|
| `crates/lift-core/` | SSA IR foundation: `Context`, types, verifier, printer, pass manager, `ModelBuilder` |
| `crates/lift-ast/` | Lexer, parser, AST, IR builder for `.lif` source files |
| `crates/lift-tensor/` | Tensor dialect (~110 ops), shape inference, FLOP counting |
| `crates/lift-quantum/` | Quantum dialect (50+ gates), noise models, topology, QEC |
| `crates/lift-hybrid/` | Hybrid dialect (21 ops), encoding strategies, gradient methods |
| `crates/lift-opt/` | 11 optimisation passes |
| `crates/lift-sim/` | Static analysis, cost models, energy estimation |
| `crates/lift-predict/` | Roofline and quantum performance prediction |
| `crates/lift-import/` | ONNX / PyTorch FX / OpenQASM importers |
| `crates/lift-export/` | LLVM IR / ONNX / OpenQASM exporters |
| `crates/lift-config/` | `.lith` configuration parser |
| `crates/lift-cli/` | `lift` command-line interface |
| `crates/lift-codegen/` | Programmatic model generation |
| `lift-test/` | Hybrid AI+Quantum integration test (medical imaging) |
| `examples/` | Hand-written and generated models, configs, validation script |

## Development Workflow

1. **Fork** the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Verify everything before pushing:

```bash
cargo fmt --all --check        # formatting
cargo clippy --all-targets -- -D warnings   # linting (warnings are errors)
cargo test --workspace        # tests
bash examples/validate_all.sh # end-to-end pipeline validation
```

4. Open a pull request. CI runs all four checks automatically.

## Guidelines

- **Keep changes minimal** and focused on one concern.
- **Preserve SSA semantics**: every value defined once, qubit linearity enforced.
- **Update documentation** when behaviour changes: `README.md`, `DIALECTS.md`,
  `CAPABILITIES.md`, and relevant crate docs.
- **Add tests** for new operations, passes, or analyses.
- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`.

## Where to Help

See [`CAPABILITIES.md`](CAPABILITIES.md) for the honest gap analysis and the
roadmap (importers, real LLVM lowering, state-vector simulator, gate
decomposition, SABRE routing, PyO3 bindings, etc.).

## Questions

Open an issue with the `question` label, or a discussion in the repository.
