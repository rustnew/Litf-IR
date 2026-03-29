# LIFT — Complete Implementation Plan

**Honest timeline: 24 months**
**Team: 2 → 4 → 8 engineers**
**Target: v1.0 public release, Q4 2026**

All corrections from design review are incorporated: realistic timelines,
realistic team, Python bindings from Phase 0, C API from Phase 0,
CI/CD from Week 1, security review as explicit milestone, maintenance plan included.

---

## Timeline at a Glance

```
  Month   1– 2    Phase 0  LIFT-CORE + Python scaffold + C API
  Month   2– 5    Phase 1  LIFT-TENSOR (AI dialect)
  Month   4– 9    Phase 2  LIFT-QUANTUM (two sub-phases)
  Month   8–12    Phase 3  LIFT-HYBRID (fusion dialect)
  Month  10–14    Phase 4  Simulation + Prediction engine
  Month  12–18    Phase 5  Backends + Importers
  Month  16–21    Phase 6  Tooling + Observability
  Month  20–24    Phase 7  Documentation + v1.0 public release

  Phases overlap intentionally.
  Backend work begins as soon as the dialect reaches alpha.
  Data collection for the GNN predictor starts in Phase 1.
```

---

## Team Scaling

```
  Months  1– 6   MVP Phase      2 engineers
    Engineer A:  Compiler (Core, AST, passes, LLVM backend)
    Engineer B:  AI + Quantum (Tensor dialect, Quantum design)

  Months  6–14   Alpha Phase    4 engineers
    + Engineer C: ML / prediction engine + benchmarks
    + Engineer D: Infrastructure (CI/CD, Python bindings, CLI)

  Months 14–24   Release Phase  8 engineers
    + Engineer E: Quantum (noise models, QPU backends, ZNE)
    + Engineer F: Hybrid dialect + joint gradient
    + Engineer G: Documentation, tutorials, community
    + Engineer H: Security review, performance, deployment
```

---

## CI/CD Pipeline — Set Up Week 1, Never Disabled

```yaml
# .github/workflows/ci.yml

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all --all-features
      - run: cargo clippy --all -- -D warnings
      - run: cargo fmt --all -- --check
      - run: cargo audit

  coverage:
    runs-on: ubuntu-latest
    steps:
      - run: cargo llvm-cov --all-features --lcov --output-path lcov.info
      - uses: codecov/codecov-action@v3
      # Fail if coverage drops below 75%

  integration:
    runs-on: [self-hosted, gpu]
    steps:
      - run: cargo test --features cuda
      - run: python tests/run_quantum_sim.py

  bench:
    runs-on: [self-hosted, gpu]
    if: github.ref == 'refs/heads/main'
    steps:
      - run: cargo bench -- --output benchmarks.json
      - run: python scripts/check_regression.py benchmarks.json
      # Fail if any benchmark regresses > 10%

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: [test, integration]
    steps:
      - run: cargo publish --package lift-core
      - run: docker build -t lift:${{ github.ref_name }} .
      - run: docker push ghcr.io/lift-framework/lift
```

**Nightly runs:**
- Full benchmark suite on GPU cluster
- Quantum regression tests (IBM simulator)
- Retrain prediction model with accumulated traces

---

## Phase 0 — LIFT-CORE (Weeks 1–8)

**Goal:** Correct, well-tested SSA IR that all dialects can build on.
**Team:** 2 engineers.

### Milestone 0.1 — IR Data Structures (Weeks 1–3)

```
[ ] Context  — SlotMap arena for all IR objects (O(1) lookup)
[ ] ValueData — TypeId + name + DefSite (OpResult or BlockArg)
[ ] OperationData — name, dialect, inputs, results, attrs, regions, location
[ ] BlockData — operations list + block arguments
[ ] RegionData — blocks list + entry block pointer
[ ] FunctionData — name + signature + region
[ ] ModuleData — name + functions + globals + dialect registry
[ ] TypeInterner — structural equality = pointer equality
[ ] StringInterner — deduplicated string storage
[ ] Attributes — typed key-value map for compile-time constants
[ ] Location — file + line + column for error messages
```

**Acceptance:** 3-operation module constructs, no memory leaks (ASAN), no panics on valid inputs.

---

### Milestone 0.2 — Parser and Printer (Weeks 2–5)

```
[ ] Lexer  (hand-written — simpler, better errors than generated)
      Tokens: ident, integer, float, string, punctuation, keywords
      Error recovery: continue after bad token, collect all errors
[ ] Parser (recursive descent)
      module, func, block, operation, type, value, attribute
[ ] AST → IR Builder
[ ] IR Printer  (human-readable .lif from IR)
[ ] Round-trip test: parse → build → print → re-parse → compare
[ ] Error messages: file + line + column + suggestion text
```

**Acceptance:** 30 hand-written .lif files parse clean. Round-trip passes all 30. Every error has a line number.

---

### Milestone 0.3 — Core Passes + Verifier (Weeks 4–7)

```
[ ] IR Verifier
      SSA property (every value defined once)
      Dominance (uses dominated by definitions)
      Type consistency (inputs/outputs match declared types)
      Well-formedness (no dangling SlotMap keys)

[ ] ConstantFoldingPass  — evaluate ops on constant inputs
[ ] DeadCodeEliminationPass  — remove unreachable ops/functions
[ ] CanonicalisationPass  — normalise patterns (add(x,0) → x, etc.)

[ ] PassManager
      Sequential runner with analysis cache
      Budget-aware rollback (rolls back passes that worsen predictions)
```

**Acceptance:** Verifier catches all 20+ intentionally malformed test IRs. Each pass has ≥ 15 unit tests. PassManager runs without panic on 100 random modules.

---

### Milestone 0.4 — Python Bindings Scaffold (Weeks 6–8)

**Done in Phase 0 — not deferred. Researchers use Python.**

```
[ ] PyO3 crate setup (lift-python)
[ ] Maturin build configuration and wheel packaging
[ ] Python type stubs (.pyi) — pass mypy --strict
[ ] Expose: lift.parse(source: str) → Module
[ ] Expose: lift.verify(module) → bool
[ ] Expose: lift.print(module) → str
[ ] Expose: lift.analyse(module) → AnalysisReport
[ ] Published to TestPyPI: pip install lift works
```

---

### Milestone 0.5 — C API (Weeks 7–8)

```
[ ] lift.h public header
[ ] lift_context_new / lift_context_free
[ ] lift_parse(ctx, source, len) → Module*
[ ] lift_module_free(module)
[ ] lift_verify(module) → int
[ ] lift_print(module) → char*    (caller frees)
[ ] lift_analyse_json(module) → char*  (caller frees)
[ ] Valgrind: no leaks through C API
```

---

## Phase 1 — LIFT-TENSOR (Weeks 5–18)

**Goal:** Correct AI dialect with working LLVM CPU backend.
**Team:** 2 engineers.

### Milestone 1.1 — Tensor Type System (Weeks 5–8)

```
[ ] TensorType (shape: Vec<Dimension>, dtype: DataType, layout: MemoryLayout)
[ ] AttentionTensor (batch, seq_len, num_heads, head_dim, dtype)
      Key insight: compiler KNOWS this is attention, not generic matmul
[ ] KVCache (max_seq, num_heads, head_dim, dtype, is_paged)
[ ] SparseTensor (num_experts, capacity, dtype)  — for MoE
[ ] Dimension (Constant(n), Symbolic(String), Product)
[ ] DataType (FP64, FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, INT8, INT4)
[ ] MemoryLayout (Contiguous, NCHW, NHWC, Strided, Tiled)
[ ] Shape inference rules for every operation
[ ] Unit tests: 20+ type system tests
```

---

### Milestone 1.2 — Core AI Operations (Weeks 7–11)

For each operation, implement: shape inference, type checking, FLOP count, memory footprint, printer, parser, ≥ 10 unit tests.

```
ARITHMETIC
[ ] tensor.add / mul / sub / div / neg
[ ] tensor.matmul {transpose_lhs, transpose_rhs}
[ ] tensor.linear (%x, %W, %b)  — fused matmul+bias
[ ] tensor.conv2d {stride, padding, dilation, groups}
[ ] tensor.embedding

ACTIVATIONS
[ ] tensor.relu / gelu / silu / sigmoid / softmax {dim}

NORMALISATION
[ ] tensor.layernorm / rmsnorm / batchnorm

SHAPE
[ ] tensor.reshape / transpose / concat / split / gather / scatter

CONSTANTS
[ ] tensor.constant / tensor.zeros / tensor.ones
```

---

### Milestone 1.3 — Attention and LLM Operations (Weeks 10–14)

```
[ ] tensor.attention {implementation, causal, scale}
      Standard | FlashAttentionV2 | FlashAttentionV3 | PagedAttention | SDPA
[ ] tensor.paged_attention {block_tables, context_len}
[ ] tensor.moe_dispatch + tensor.moe_combine
[ ] tensor.quantize / tensor.dequantize
[ ] tensor.checkpoint {fn}  — gradient recomputation
[ ] tensor.offload {location, prefetch}  — CPU/SSD offloading for 10T+ models
[ ] Gradient operations for all the above
```

---

### Milestone 1.4 — AI Optimisation Passes (Weeks 12–17)

```
[ ] TensorFusionPass
      Declarative pattern library (≥ 10 patterns)
      Topological matching — O(V+E×P), NOT Ullmann O(n!)
      Profitability check: only fuse when single_use(intermediates)
      Tests: 15+ including negative cases (must not fuse when not safe)

[ ] FlashAttentionPass
      Detect tensor.attention {implementation=Standard}
      Condition: seq_len > 512 AND GPU target
      Replace with FlashAttentionV2 or V3 based on arch (sm_80 vs sm_90)

[ ] KVCachePass
      Detect attention in inference mode (no gradient)
      Insert paged attention + KV cache allocation

[ ] QuantizationPass
      Dynamic INT8 (no calibration data required)
      Static INT8 (requires calibration_dataset in .lith)
      FP8 (sm_90 / H100 only)
      Per-channel or per-tensor

[ ] ParallelismPass
      Data / Tensor / Pipeline / Sequence parallelism
      Insert explicit split / allreduce / send / receive ops

[ ] MemoryPlanningPass
      Liveness analysis → buffer reuse
      Memory pool creation
```

**Correctness validation:** Run all 5 000 reference programmes through each pass. Compare outputs against PyTorch within 1e-5 tolerance.

---

### Milestone 1.5 — LLVM CPU Backend (Weeks 14–18)

```
[ ] LLVM IR emitter for all tensor ops
[ ] AVX-512 SIMD hints for contiguous tensor ops
[ ] OpenMP pragmas for element-wise ops
[ ] Shared library (.so) and native executable output
[ ] End-to-end test: LLaMA 7B inference compiles + output within 1e-4 of PyTorch
```

---

## Phase 2 — LIFT-QUANTUM (Weeks 15–36)

Split into two sub-phases due to complexity.

### Phase 2a — Basic Quantum (Weeks 15–24)

#### Milestone 2a.1 — Quantum Types + Linearity (Weeks 15–18)

```
[ ] Qubit type  — linear (consumed exactly once, enforced by verifier)
[ ] PhysicalQubit type  — id, T1, T2, frequency, fidelity
[ ] ClassicalBit type
[ ] QuantumState type  — StateVector | DensityMatrix | MPS | Stabiliser
[ ] Hamiltonian type  — Vec<PauliTerm>

LINEARITY CHECKER (integrated into verifier)
[ ] Track consumed: HashSet<ValueKey> per block
[ ] Report error if qubit key appears twice as input
[ ] Report error if branch arms consume different qubit sets
[ ] Unit tests: 10 valid programmes + 10 that must fail
```

#### Milestone 2a.2 — Gate Operations (Weeks 17–21)

```
SINGLE-QUBIT
[ ] H, X, Y, Z, S, Sdg, T, Tdg, SX
[ ] RX(θ), RY(θ), RZ(θ), P(λ), U1, U2, U3

TWO-QUBIT
[ ] CX, CZ, CY, SWAP, iSWAP, ECR
[ ] RZX(θ), XX(θ), YY(θ), ZZ(θ)

THREE-QUBIT
[ ] CCX (Toffoli), CSWAP (Fredkin)

CONTROL
[ ] quantum.measure {basis}  → bit
[ ] quantum.measure_all      → tensor<n×bit>
[ ] quantum.reset             → qubit  (recycles linearly)
[ ] quantum.barrier           (no-optimise fence)
[ ] quantum.init()            → qubit

PARAMETRISED
[ ] quantum.param_gate {type, qubits, params}  — for VQE/QAOA
```

#### Milestone 2a.3 — State Vector Simulator + Basic OpenQASM (Weeks 20–24)

```
[ ] CPU state vector (up to 28 qubits, exact)
[ ] GPU state vector (up to 35 qubits, via cuStateVec or custom)
[ ] Basic OpenQASM 3 emitter (all gates)
      IBM basis decomposition: {RZ, SX, X, CX}
      Rigetti basis decomposition: {RZ, RX, CZ}
```

### Phase 2b — Advanced Quantum (Weeks 22–36)

#### Milestone 2b.1 — Noise Models + Density Matrix (Weeks 22–27)

```
[ ] IBM device calibration loader (JSON from IBM Quantum API)
[ ] Rigetti device calibration loader
[ ] Noise channels: depolarising, amplitude damping, Pauli error
[ ] Crosstalk model (ZZ coupling between neighbours)
[ ] Readout error matrix
[ ] Noise propagation analysis (accumulated error through circuit)
[ ] Density matrix simulator (up to 20 qubits, includes noise)
[ ] MPS simulator (up to 100 qubits, low-entanglement circuits)
[ ] Monte Carlo noise simulation (trajectory sampling)
```

#### Milestone 2b.2 — Layout Mapping + Quantum Passes (Weeks 26–33)

```
[ ] QuantumTopology (coupling map + gate fidelity + gate time)
[ ] SABRE routing — standard variant
[ ] SABRE routing — noise-aware variant (fidelity-weighted scoring)
[ ] A* exact routing for small circuits (≤ 10 qubits)
[ ] SWAP insertion verification (simulate before+after, compare within 1e-5)

[ ] GateCancellationPass  — identities + commutation table
[ ] RotationMergingPass   — Rz(a)·Rz(b) = Rz(a+b) with wrapping
[ ] GateDecompositionPass — hardware-native basis sets
[ ] TwoQubitWeylDecompositionPass — Cartan/KAK, ≤ 3 CX gates
[ ] TemplateMatchingPass  — known-optimal circuit fragments
```

#### Milestone 2b.3 — Error Mitigation Passes (Weeks 30–36)

```
[ ] ZNE (Zero Noise Extrapolation)
      Gate folding: G → G G† G (3× noise factor)
      Richardson extrapolation (auto-order: linear/quadratic/cubic)
      R² validation: warn if fit quality < 0.95

[ ] Readout error mitigation
      Calibration circuits (all-zeros, all-ones)
      Correction matrix inversion

[ ] Dynamical Decoupling
      Detect idle periods > T2/10
      Insert XY-4 sequences, respect hardware timing constraints
```

---

## Phase 3 — LIFT-HYBRID (Weeks 28–42)

### Milestone 3.1 — Encoding Operations (Weeks 28–33)

```
[ ] hybrid.amplitude_encode {normalize}
      tensor<N×f32> → log₂(N) qubits
      Gate sequence to initialise the state vector

[ ] hybrid.angle_encode {gate}
      tensor<N×f32> → N qubits
      Domain mapping: float → [0, 2π]

[ ] hybrid.basis_encode
      tensor<N×i32> → N qubits

[ ] hybrid.decode (measurement → tensor)
      Expectation value, sampling statistics
```

### Milestone 3.2 — Hybrid Operations (Weeks 32–39)

```
[ ] hybrid.angle_encode_forward
      Most common QNN pattern: encode + parameterised circuit combined

[ ] hybrid.measure_with_ml
      Quantum measurement → classical network post-processing

[ ] hybrid.parameter_shift_gradient
      d⟨O⟩/dθ = [⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)] / 2
      Batches 2P circuit evaluations efficiently

[ ] hybrid.joint_optimisation
      Classical + quantum params, single optimiser step
      Supports Adam, COBYLA, SPSA

[ ] hybrid.cosimulation {interface}
      GPU-side + QPU-side co-execution
      Synchronisation and data transfer management
```

### Milestone 3.3 — Hybrid Passes (Weeks 37–42)

```
[ ] HybridFusionPass  — merge classical ops with quantum measurement
[ ] ParameterShiftPass  — expand gradients into 2P evaluations
[ ] EncodingOptimisationPass  — amplitude vs angle tradeoff
[ ] ShotOptimisationPass  — minimum shots for target precision
```

---

## Phase 4 — Simulation + Prediction (Weeks 32–46)

### Milestone 4.1 — Static Analysis Engine (Weeks 32–37)

```
[ ] Full shape propagation (all tensor + quantum ops)
[ ] Full FLOP counter (per-op formulae, per-module total)
[ ] Memory liveness analysis (peak memory computation)
[ ] Bandwidth pressure estimator
[ ] Circuit analyser: depth, gate counts, T1/T2 decoherence risk
[ ] Energy model (per-op energy table × count + infrastructure)
[ ] Carbon estimate (energy × grid intensity, region configurable)
[ ] HTML + JSON simulation report generator
```

### Milestone 4.2 — GNN Prediction Model (Weeks 35–44)

**Data collection starts in Phase 1, runs continuously.**

```
DATA PIPELINE (start Week 5, run until Week 38)
[ ] Automatic benchmark runner (200+ programmes × 5+ hardware configs)
[ ] Trace format: (IR graph JSON, hw_spec JSON, latency_ms, memory_gb)
[ ] Target: 100K examples before training

MODEL (Weeks 38–42)
[ ] Graph feature extractor (IR → node/edge matrices)
[ ] GNN: NodeEncoder(128) + EdgeEncoder(64) + 6×GatedGraphConv(128)
       + GlobalAttentionPool + HWEncoder(64)
       + LatencyHead + MemoryHead + FidelityHead (quantum)
[ ] Training pipeline (PyTorch, export to ONNX)
[ ] Rust ONNX inference engine (prediction < 100ms)
[ ] Analytical roofline fallback (confidence < 0.70)
[ ] Confidence scoring

VALIDATION (Weeks 42–44)
[ ] Hold-out set: 10K examples
[ ] Acceptance: median error < 15% on held-out set
[ ] OOD test: predict on hardware not in training set
```

### Milestone 4.3 — Budget Enforcement (Weeks 43–46)

```
[ ] Budget constraint parsing from .lith
[ ] Budget checker comparing predictions to constraints
[ ] Actionable error messages (what violated, by how much)
[ ] Suggestion engine (top-3 passes that would resolve the violation)
```

---

## Phase 5 — Backends + Interoperability (Weeks 38–56)

### Milestone 5.1 — CUDA Backend (Weeks 38–44)

```
[ ] PTX code generation framework
[ ] tensor.matmul → cuBLAS (FP32, FP16, INT8)
[ ] tensor.flash_attention → custom FlashAttention v2/v3 kernel
[ ] tensor.conv2d → cuDNN
[ ] tensor.quantize/dequantize → INT8 CUDA kernels
[ ] Tensor Core utilisation (FP16, BF16, INT8, FP8)
[ ] Memory coalescing enforcement
[ ] NCCL integration for multi-GPU allreduce
[ ] CUDA graph capture for inference
[ ] E2E test: LLaMA 7B on H100, output within 1e-4, within 10% of TensorRT
```

### Milestone 5.2 — OpenQASM 3 Full Backend (Weeks 42–47)

```
[ ] Complete OpenQASM 3.0 emitter
[ ] IBM Qiskit Runtime: create job, poll, retrieve results
[ ] AWS Braket submission (Rigetti, IonQ)
[ ] Pulse-level lowering for IBM (DRAG pulse shaping)
[ ] E2E test: VQE H₂ on IBM Kyoto, energy within 1 mHa of literature
```

### Milestone 5.3 — Importers (Weeks 44–52)

| Importer | Target | Notes |
|----------|--------|-------|
| PyTorch FX | 100% | Complete from current 80% |
| ONNX | opset 19 | Build from scratch |
| Qiskit QuantumCircuit | 100% | Including NoiseModel import |
| Cirq Circuit | 100% | Standard gates |
| OpenQASM 3 | 100% | Complete from current 60% |

### Milestone 5.4 — .lith Parser (Weeks 48–54)

```
[ ] Complete grammar (all sections and fields)
[ ] Environment variable substitution: ${VAR_NAME}
[ ] File inclusion: include "./base.lith"
[ ] Config inheritance: extends = "base.lith"
[ ] Conditional blocks: if target.type == "qpu" { ... }
[ ] Enum validation (all field values checked against allowed set)
[ ] Cross-section consistency checks
[ ] Helpful errors (line + column + suggestion)
[ ] Auto-generated reference documentation
```

---

## Phase 6 — Tooling + Observability (Weeks 52–62)

### Milestone 6.1 — Full CLI (Weeks 52–56)

```
lift compile  <file.lif> [--config <.lith>] [--target cuda|qasm|llvm]
lift simulate <file.lif> [--config <.lith>] [--report <html>]
lift predict  <file.lif> [--hardware h100|ibm_kyoto|...]
lift optimise <file.lif> [--passes <p1,p2,...>] [--output <file.lif>]
lift convert  [--from pytorch|onnx|qiskit|qasm] [--to lift] <input>
lift verify   <file.lif>
lift analyse  <file.lif>
lift info     <file.lif>
```

### Milestone 6.2 — Observability (Weeks 54–60)

```
[ ] Structured JSON logging (tracing crate) — every pass logged
[ ] Prometheus metrics endpoint (/metrics)
      compilation_duration_seconds histogram per pass
      prediction_error_ratio (predicted vs actual)
[ ] Interactive web dashboard (port 8081)
      Compilation trace timeline
      Computation graph viewer (interactive)
      Circuit diagram viewer
      IR diff viewer (before/after pass)
[ ] Compilation replay (record inputs + config for debugging)
```

### Milestone 6.3 — Auto-Tuning (Weeks 58–62)

```
[ ] Pass ordering search (Bayesian optimisation, 50-iteration budget)
[ ] RL-based pass selector (GNN agent, trained on compilation traces)
[ ] A/B testing infrastructure (10% traffic split, compare real perf)
```

---

## Phase 7 — Documentation + v1.0 (Weeks 60–96)

### Milestone 7.1 — Documentation (Weeks 60–70)

```
[ ] API documentation (rustdoc, all public items)
[ ] Language reference (.lif — all ops, types, dialects)
[ ] Configuration reference (.lith — all sections and fields)
[ ] Getting Started Guide (30 min: install → hello world → real model)

TUTORIALS
[ ] Tutorial 1: LLM Inference Optimisation
      LLaMA 7B → FlashAttention → KV cache → INT8 → benchmark
[ ] Tutorial 2: VQE for Hydrogen Molecule
      H₂ VQE → LIFT-QUANTUM → noise model → ZNE → IBM Kyoto
[ ] Tutorial 3: QNN Image Classifier
      MNIST → hybrid QNN → angle encoding → joint gradient → train
[ ] Tutorial 4: Writing a Custom Pass
      Implement simple fusion → register → use from .lith
```

### Milestone 7.2 — Benchmarks (Weeks 68–80)

| Category | Programmes | Compare against |
|----------|-----------|----------------|
| AI | BERT-Large, LLaMA 7B, ResNet-50, ViT-Large | PyTorch + torch.compile + TensorRT |
| Quantum | QAOA MaxCut, VQE H₂O, QV=128, random circuits | Qiskit transpile + pytket |
| Hybrid | QNN MNIST, QAOA+post-process | PennyLane |
| Compile time | All models above | torch.compile, Qiskit transpile |

### Milestone 7.3 — Security Review (Weeks 76–84)

```
[ ] Fuzzing: 1M+ random .lif inputs to parser
[ ] Fuzzing: 100K+ random .lith configs
[ ] cargo audit automated in CI
[ ] cargo-deny for license + advisory policy
[ ] No unsafe in core crates (MIRI verification)
[ ] Sandboxed compilation (seccomp)
[ ] Security disclosure policy: security@lift-framework.org
[ ] GPG signing for all releases
```

### Milestone 7.4 — Public v1.0 (Weeks 84–96)

```
[ ] GitHub: lift-framework/lift (MIT, README, CONTRIBUTING, CODE_OF_CONDUCT)
[ ] crates.io: lift-core, lift-tensor, lift-quantum, lift-hybrid, lift-cli
[ ] PyPI: lift (via Maturin)
[ ] Docker: lift:latest (all backends pre-installed)
[ ] arXiv: "LIFT: A Unified IR for AI and Quantum Computing"
[ ] Blog post (3 000 words)
[ ] Announcements: HN, r/MachineLearning, r/QuantumComputing
[ ] Discord server, GitHub Discussions, issue templates
```

---

## Hardware Requirements

### Development

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× A100 40 GB | 2× H100 80 GB |
| CPU | 16-core | 32-core server |
| RAM | 128 GB | 256 GB |
| QPU access | 10 QPU-hours/month | 30 QPU-hours/month |

### CI/CD

| Resource | Purpose |
|----------|---------|
| 1× A100 runner | GPU integration tests |
| Quantum sim server (64-core, 256 GB) | Large circuit simulations |
| IBM Quantum + AWS Braket credits | Real QPU tests |

### GNN Training (Phase 4)

| Resource | Purpose |
|----------|---------|
| 2 000 GPU-hours | Training data collection across hardware |
| 50 GPU-hours | Model training and validation |
| 20 QPU-hours | Quantum calibration data for noise models |

---

## Success Criteria

### At 6 Months (End of Phase 1)

```
[ ] 500+ passing tests across lift-core and lift-tensor
[ ] 10+ AI models compile to LLVM and produce correct output
[ ] Compilation time < 10 seconds for 7B-parameter model (LLVM target)
[ ] Python API functional: import lift; lift.analyse() works
[ ] Zero memory leaks (ASAN + Valgrind)
[ ] CI green on every commit for 30 consecutive days
[ ] Code coverage ≥ 75%
```

### At 12 Months (End of Phase 2b)

```
[ ] Quantum circuits up to 100 qubits simulate correctly (vs Qiskit)
[ ] SABRE layout produces correct results on IBM coupling maps
[ ] ZNE pass improves fidelity > 5× on noisy simulations
[ ] Import Qiskit circuits and compile to OpenQASM 3
[ ] E2E: VQE H₂ on IBM Kyoto, correct ground state energy
```

### At 18 Months (End of Phase 5)

```
[ ] LLaMA 7B inference on H100 within 10% of TensorRT
[ ] QNN MNIST trains end-to-end with joint gradients
[ ] .lith parser validates complete configs with helpful errors
[ ] All 5 importers functional (PyTorch FX, ONNX, Qiskit, Cirq, OpenQASM3)
[ ] Code coverage ≥ 80%
```

### At 24 Months (v1.0 Release)

```
[ ] arXiv preprint submitted
[ ] 100+ GitHub stars in first 30 days
[ ] 5+ external contributors
[ ] Benchmark results published and competitive
[ ] 3+ tutorials published
[ ] 2+ external research groups using LIFT
```

---

## Risk Register

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| Linear types in branches complex to implement | High | Medium | Prototype region analysis in Phase 0; descope to subset without branches if needed |
| GNN predictor does not generalise | Medium | High | Analytical fallback; ensemble; more training data |
| QPU access limits quantum testing | High | Medium | Use simulators for 95% of tests; batch real QPU monthly |
| CUDA backend performance gap vs TensorRT | Medium | Medium | Use cuBLAS/cuDNN for critical kernels |
| Timeline slips | High | Medium | 24-month buffer built in; descope features before delaying |
| Noise composition after fusion incorrect | Medium | High | Depolarising approx v1.0; Kraus v1.1; flagged in docs |
| Community adoption too slow | Medium | High | Partner with 2 research groups early; active Discord |
| Key engineer departure | Low | High | Document everything; pair programming; knowledge transfer |

---

## Maintenance Plan

### Release Cadence

| Release type | Frequency | Supported for |
|-------------|-----------|--------------|
| Nightly | Daily | Not supported |
| Alpha | Weekly | 2 weeks |
| Beta | Monthly | 3 months |
| Stable | Quarterly | 18 months |
| LTS | Every 2 years | 3 years |

### Deprecation Policy

1. Announce deprecation in release notes with migration guide.
2. Compiler warning when deprecated feature is used.
3. Remove after one full stable release cycle (minimum 3 months notice).

### Community Health Targets

| Metric | Target at v1.0 | Target at v2.0 |
|--------|---------------|---------------|
| GitHub stars | 100 | 1 000 |
| External contributors | 5 | 30 |
| Research groups using LIFT | 2 | 20 |
| Issue response time | < 48h | < 24h |
| PR merge time | < 2 weeks | < 1 week |

---

*LIFT — Built correctly, documented honestly, shipped on time.*

*The first unified IR for AI and Quantum Computing.*