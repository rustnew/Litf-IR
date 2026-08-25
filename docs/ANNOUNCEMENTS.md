# LIFT — Community Announcements

Ready-to-post announcement texts for each community channel. Each is tuned to
the platform's tone and audience. Replace the placeholder links if needed.

**Key facts** (verified):
- 13 crates on crates.io (v0.4.6), docs on docs.rs
- 110 tensor ops, 48 quantum gates, 21 hybrid ops
- 13 optimisation passes, O0–O3 pipelines
- 3 backends: LLVM IR, ONNX (opset 21), OpenQASM 3.0
- Docs book: https://rustnew.github.io/Lift/

---

## Reddit — r/rust (showcase)

**Title:** LIFT — a unified compiler framework for AI and quantum computing in Rust

**Body:**

I've been building **LIFT**, a compiler framework that treats AI and quantum
computing as one problem instead of two.

The core idea: a single SSA intermediate representation where tensor ops,
quantum gates, and classical-quantum hybrids are equal citizens. So you can
optimise a hybrid VQE/QAOA workload and a transformer model in the same
pipeline, with the same passes.

What it does today:
- **110 tensor ops** — attention (Flash/Paged/GQA), MoE, quantisation, GNN, diffusion
- **48 quantum gates** — with noise models, Kraus channels, QEC codes
- **13 optimisation passes** — tensor fusion, FlashAttention replacement, gate
  cancellation, noise-aware scheduling, qubit routing (SWAP + BFS), gate decomposition
- **O0–O3 pipelines** with per-pass control
- **3 backends** — LLVM IR, ONNX (opset 21), OpenQASM 3.0
- **Cost modelling** — FLOPs, memory, energy, roofline prediction *before* hardware runs

It's published as 13 crates on crates.io, with full docs.

- GitHub: https://github.com/rustnew/Lift
- crates.io: https://crates.io/crates/lift-core
- docs.rs: https://docs.rs/lift-core
- Docs book: https://rustnew.github.io/Lift/

Happy to hear feedback — especially from anyone working on MLIR, TVM, or
quantum compilers. The roadmap (simulator, importers, real LLVM lowering) is
open for contributions.

---

## Reddit — r/QuantumComputing

**Title:** LIFT — a Rust compiler that unifies AI tensor and quantum circuit compilation

**Body:**

Sharing a project I've been working on: **LIFT**, a compiler framework with a
single SSA IR that spans tensor operations *and* quantum gates.

For the quantum side, it includes:
- 48 quantum gates with noise models, Kraus channels, and QEC codes
- Noise-aware scheduling — the compiler reasons about T1/T2/fidelity at every stage
- Linear qubit types — the no-cloning theorem is enforced at compile time
- Qubit layout mapping and real qubit routing (SWAP + BFS)
- Hardware-native gate decomposition (IBM, Rigetti, IonQ, Quantinuum)
- OpenQASM 3.0 export

The differentiator: because AI tensors and quantum gates share one IR, hybrid
classical-quantum workloads (VQE, QAOA, quantum chemistry) can be optimised
jointly with the classical parts.

- GitHub: https://github.com/rustnew/Lift
- crates.io: https://crates.io/crates/lift-core
- Docs: https://rustnew.github.io/Lift/

The state-vector simulator and importers (Qiskit, OpenQASM) are on the roadmap.

---

## Reddit — r/MachineLearning

**Title:** [P] LIFT — a Rust compiler framework for AI and quantum workloads

**Body:**

I've been working on **LIFT**, a compiler framework that unifies AI and quantum
computation under one SSA intermediate representation.

The ML-relevant parts:
- 110 tensor ops including attention (Flash/Paged/GQA), MoE, quantisation, GNN, diffusion
- 13 optimisation passes including tensor fusion and FlashAttention replacement
- Cost modelling: FLOPs, peak memory, energy, and roofline prediction computed
  before hardware runs — budget violations halt compilation with suggestions
- ONNX (opset 21) export for PyTorch/TensorFlow/TensorRT interop

It's written in Rust and published as 13 crates.

- GitHub: https://github.com/rustnew/Lift
- crates.io: https://crates.io/crates/lift-core
- Docs: https://rustnew.github.io/Lift/

The tensor interpreter (numpy-like execution) and real LLVM lowering are on
the roadmap. Feedback welcome.

---

## Hacker News — Show HN

**Title:** Show HN: LIFT — a unified compiler for AI and quantum computing

**Body:**

I've been working on a compiler framework that treats AI and quantum computing
as a single problem. LIFT uses one SSA intermediate representation where tensor
ops, quantum gates, and classical-quantum hybrids are all first-class.

Why this matters: hybrid workloads (VQE, QAOA, quantum chemistry) need both
classical and quantum compilation, but today they live in separate toolchains
with separate IRs. LIFT lets you optimise them together.

Current state:
- 110 tensor ops, 48 quantum gates, 21 hybrid ops
- 13 optimisation passes, O0–O3 pipelines
- Noise-aware scheduling + linear qubit types (no-cloning enforced at compile time)
- Cost modelling before hardware runs (FLOPs, memory, energy, roofline)
- LLVM IR / ONNX / OpenQASM 3.0 backends
- 13 crates on crates.io, MIT licensed

Written in Rust. Docs: https://rustnew.github.io/Lift/

The roadmap (state-vector simulator, importers, real LLVM lowering) is open.
Would love feedback from compiler folks — especially anyone who's worked with
MLIR or quantum transpilers.

---

## This Week in Rust — submission

**Title:** LIFT: a unified compiler framework for AI and quantum computing

**Body:**

[LIFT](https://github.com/rustnew/Lift) is a Rust compiler framework with a
single SSA intermediate representation spanning tensor operations, quantum
gates, and classical-quantum hybrids. It ships 13 optimisation passes, O0–O3
pipelines, cost modelling, and LLVM IR / ONNX / OpenQASM 3.0 backends across 13
crates on [crates.io](https://crates.io/crates/lift-core).

---

## users.rust-lang.org — Announcements

**Title:** LIFT — a unified compiler framework for AI and quantum computing

**Body:**

I'm announcing **LIFT**, a Rust compiler framework that unifies AI and quantum
computation under a single SSA intermediate representation.

Highlights:
- 110 tensor ops, 48 quantum gates, 21 hybrid ops
- 13 optimisation passes, O0–O3 pipelines
- Noise-aware scheduling and linear qubit types
- Cost modelling (FLOPs, memory, energy, roofline) before hardware runs
- LLVM IR / ONNX / OpenQASM 3.0 backends
- 13 crates on crates.io, MIT licensed

Links:
- GitHub: https://github.com/rustnew/Lift
- crates.io: https://crates.io/crates/lift-core
- docs.rs: https://docs.rs/lift-core
- Docs book: https://rustnew.github.io/Lift/

Contributions welcome — see CONTRIBUTING.md.