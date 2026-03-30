# LIFT Framework — Comprehensive Improvement Plan
## For Windsurf Integration and Automated Updates
### Based on v0.2.0 → v1.0.0 Roadmap

> **Purpose:** This document is the authoritative guide for Windsurf to understand,
> plan, and execute all improvements to the LIFT framework.
> Every section is actionable, scoped to a specific crate, and ordered by priority.

---

## Table of Contents

1. [Project Overview & Current State](#1-project-overview--current-state)
2. [Phase 1 — Semantic Depth (Months 1–3)](#2-phase-1--semantic-depth-months-13)
3. [Phase 2 — Precision & Prediction (Months 2–5)](#3-phase-2--precision--prediction-months-25)
4. [Phase 3 — Developer Experience (Months 3–6)](#4-phase-3--developer-experience-months-36)
5. [Phase 4 — Hardware Backends (Months 4–8)](#5-phase-4--hardware-backends-months-48)
6. [Phase 5 — Stability & Production (Months 6–12)](#6-phase-5--stability--production-months-612)
7. [Cross-Cutting Concerns](#7-cross-cutting-concerns)
8. [File-by-File Modification Map](#8-file-by-file-modification-map)
9. [Test Strategy](#9-test-strategy)
10. [Priority Matrix](#10-priority-matrix)

---

## 1. Project Overview & Current State

### 1.1 What Exists (v0.2.0) — Do NOT Break

```
lift/
├── crates/
│   ├── lift-core/        ✅ STABLE — SSA IR, types, verifier, printer, pass manager
│   ├── lift-ast/         ✅ STABLE — Lexer, parser, builder for .lif files
│   ├── lift-tensor/      🚧 ACTIVE — 40+ ops, FLOPs, memory (needs more ops)
│   ├── lift-quantum/     🚧 ACTIVE — 28+ gates, noise, topology (needs Kraus, QEC)
│   ├── lift-hybrid/      🚧 ACTIVE — 11 ops (needs more encoding strategies)
│   ├── lift-sim/         🚧 ACTIVE — FLOPs, memory, noise sim (needs GNN)
│   ├── lift-predict/     📐 DESIGN — Roofline only (needs GNN predictor)
│   ├── lift-opt/         🚧 ACTIVE — 5 passes (needs 8+ more)
│   ├── lift-import/      🚧 ACTIVE — PyTorch FX ~80%, ONNX, QASM ~60%
│   ├── lift-export/      🚧 ACTIVE — LLVM ~70%, QASM ~40% (needs CUDA)
│   ├── lift-config/      🚧 ACTIVE — .lith ~60% (needs inheritance, validation)
│   ├── lift-cli/         🚧 ACTIVE — 6 commands (needs compile, simulate, predict)
│   └── lift-tests/       ✅ 165 tests — KEEP ALL PASSING
```

### 1.2 Invariants — Never Violate

- All 165 existing tests must remain GREEN at every commit.
- SSA single-definition invariant must hold across all new operations.
- Linear qubit type rule: each qubit consumed exactly once.
- All new ops must implement `flop_count()` and `memory_bytes()`.
- All new passes must be semantics-preserving (validated by differential testing).

### 1.3 Key Numbers to Beat

| Metric | v0.2.0 | Target v1.0 |
|--------|--------|-------------|
| Tests | 165 (100%) | 1000+ (100%) |
| Tensor ops | 40+ | 80+ |
| Quantum gates | 28+ | 50+ |
| Optimisation passes | 5 | 15+ |
| FLOPs prediction error | < 5% | < 1% |
| Latency prediction error | unknown | < 10% |
| Fidelity prediction error | < 5% | < 2% |

---

## 2. Phase 1 — Semantic Depth (Months 1–3)

### 2.1 `lift-tensor` — Extend to Full AI Coverage

**File:** `crates/lift-tensor/src/ops.rs`

#### 2.1.1 New Operations to Add

```rust
// ATTENTION VARIANTS (add to TensorOp enum)
MultiHeadAttention {
    num_heads: u32,
    head_dim: u32,
    dropout: Option<f32>,
},
MultiQueryAttention {          // MQA — used in Falcon, PaLM
    num_heads: u32,
    num_kv_heads: u32,         // << num_heads
    head_dim: u32,
},
GroupedQueryAttention {        // GQA — used in LLaMA-2/3, Mistral
    num_heads: u32,
    num_groups: u32,
    head_dim: u32,
},
FlashAttention {               // IO-aware exact attention
    seq_len: u32,
    head_dim: u32,
    causal: bool,
    block_size: u32,
},
SlidingWindowAttention {       // Mistral
    window_size: u32,
},

// CONVOLUTION VARIANTS
Conv1d { in_channels: u32, out_channels: u32, kernel: u32, stride: u32, padding: u32 },
Conv3d { in_channels: u32, out_channels: u32, kernel: [u32;3], stride: [u32;3] },
ConvTranspose2d { in_channels: u32, out_channels: u32, kernel: [u32;2], stride: [u32;2] },
DepthwiseConv2d { channels: u32, kernel: [u32;2], multiplier: u32 },
DilatedConv2d { in_channels: u32, out_channels: u32, kernel: [u32;2], dilation: u32 },

// RECURRENT
LstmCell { input_size: u32, hidden_size: u32 },
GruCell  { input_size: u32, hidden_size: u32 },
RnnCell  { input_size: u32, hidden_size: u32 },

// ADVANCED MATH
Einsum  { equation: String },  // e.g. "bij,bjk->bik"
Fft     { n: Option<u32>, dim: i32 },
Ifft    { n: Option<u32>, dim: i32 },
Svd     { full_matrices: bool },
Eig     { },
Solve   { },                   // Ax = b

// POOLING
MaxPool2d  { kernel: [u32;2], stride: [u32;2], padding: [u32;2] },
AvgPool2d  { kernel: [u32;2], stride: [u32;2], padding: [u32;2] },
AdaptiveAvgPool2d { output_size: [u32;2] },
GlobalAvgPool { },

// NORMALISATION ADDITIONS
GroupNorm  { num_groups: u32, num_channels: u32 },
InstanceNorm { num_features: u32 },

// SPARSE
SparseMatmul { sparsity: f32 },
SparseEmbedding { vocab_size: u32, embed_dim: u32, sparse: bool },

// QUANTISATION (native INT4/FP8)
QuantizeInt4   { group_size: u32 },
DequantizeInt4 { group_size: u32 },
QuantizeFp8    { format: Fp8Format },   // E4M3 or E5M2
DequantizeFp8  { format: Fp8Format },

// DIFFUSION / GENERATIVE
UnetDownBlock  { in_channels: u32, out_channels: u32, time_emb_dim: u32 },
UnetUpBlock    { in_channels: u32, out_channels: u32, time_emb_dim: u32 },
CrossAttention { query_dim: u32, context_dim: u32, num_heads: u32 },
TimestepEmbedding { dim: u32 },

// GNN (Graph Neural Networks)
GnnMessagePassing { in_features: u32, out_features: u32, aggr: AggregationType },
GnnGlobalPooling  { in_features: u32, aggr: AggregationType },
```

**File:** `crates/lift-tensor/src/flops.rs` (new file)

```rust
// Exact FLOPs formulas — ALL operations must implement this trait
pub trait FlopCount {
    fn flop_count(&self, input_shapes: &[Shape]) -> u128;
}

// Key formulas:
// MatMul(M,K) x (K,N)          => 2 * M * N * K
// Conv2d                        => 2 * C_out * C_in * Kh * Kw * Oh * Ow
// Attention(B,S,H,D)            => 2 * B * H * (S^2 * D + S * D^2)  [QK + AV]
// FlashAttention                => same FLOPs, different memory
// LSTM(input_size, hidden)      => 4 * (input_size + hidden) * hidden * 2 (per step)
// Einsum                        => parse equation, compute output shape, count mults
// FFT(N)                        => 5 * N * log2(N)
// LayerNorm(N)                  => 7 * N  (mean + var + normalize + scale + shift)
// GeLU(N)                       => 8 * N  (erf approximation)
// Softmax(N)                    => 5 * N  (exp + sum + div)
```

**File:** `crates/lift-tensor/src/types.rs`

```rust
// Add semantic type annotations for dimensions
pub enum DimAnnotation {
    Batch,
    Sequence { max_len: Option<u32> },
    Heads,
    HeadDim,
    Channels,
    Spatial,
    Vocab,
    Hidden,
    Experts,
    Unknown,
}

// Annotated tensor type
pub struct AnnotatedTensorType {
    pub base: TensorType,
    pub dim_annotations: Vec<DimAnnotation>,
    pub parallel_strategy: Option<ParallelStrategy>,
}

pub enum ParallelStrategy {
    TensorParallel { axis: usize },
    SequenceParallel { axis: usize },
    ExpertParallel { num_experts: u32 },
    DataParallel,
    None,
}
```

---

### 2.2 `lift-quantum` — Complete the Gate Catalogue

**File:** `crates/lift-quantum/src/gates.rs`

#### New Gates to Add

```rust
// PARAMETRIC ADDITIONS
Rx90,                // Fixed pi/2 rotation (native on many QPUs)
Rx180,               // Fixed pi rotation
PhaseShift { theta: f64 },

// IBM NATIVE BASIS SET
EchoedCrossResonance,    // ECR (already present, verify)
ControlledPhase { theta: f64 },

// RIGETTI NATIVE BASIS SET
Cphase { theta: f64 },
XY { theta: f64 },

// ION TRAP NATIVES
GPI  { phi: f64 },       // IonQ native
GPI2 { phi: f64 },       // IonQ native
MS   { phi0: f64, phi1: f64 },  // Molmer-Sorensen

// MULTI-QUBIT (beyond 3)
MCX { num_controls: u32 },  // Multi-controlled X
MCZ { num_controls: u32 },  // Multi-controlled Z

// SPECIAL
GlobalPhase { theta: f64 },
Delay { duration_ns: f64 },          // Timing control
VirtualRZ  { theta: f64 },           // Software phase frame
IfElse { condition_bit: usize },     // Classical control
```

**File:** `crates/lift-quantum/src/noise.rs`

```rust
// KRAUS OPERATOR MODEL (upgrade from simple fidelity model)
pub struct KrausChannel {
    pub operators: Vec<ComplexMatrix>,   // Sum(K_i† K_i) = I
}

impl KrausChannel {
    // Compose two channels: K1 followed by K2
    pub fn compose(&self, other: &KrausChannel) -> KrausChannel { ... }

    // Convert to fidelity (average gate fidelity)
    pub fn average_gate_fidelity(&self) -> f64 { ... }

    // Depolarising channel: p*(I/d) + (1-p)*rho
    pub fn depolarizing(p: f64, n_qubits: u32) -> Self { ... }

    // Amplitude damping: T1 process
    pub fn amplitude_damping(gamma: f64) -> Self { ... }

    // Phase damping: T2 dephasing
    pub fn phase_damping(lambda: f64) -> Self { ... }

    // Pauli channel (efficient for Clifford simulation)
    pub fn pauli(px: f64, py: f64, pz: f64) -> Self { ... }
}

// Per-gate calibration data (loaded from hardware or JSON file)
pub struct GateCalibration {
    pub gate: QuantumGate,
    pub qubits: Vec<usize>,
    pub t1: f64,
    pub t2: f64,
    pub gate_fidelity: f64,
    pub gate_time_ns: f64,
    pub readout_fidelity: f64,
    pub crosstalk: HashMap<(usize, usize), f64>,
    pub timestamp: DateTime<Utc>,
}
```

**File:** `crates/lift-quantum/src/qec.rs` (new file)

```rust
// Quantum Error Correction support
pub enum QecCode {
    SurfaceCode { distance: u32 },
    SteaneCode,
    ShorCode,
    RepetitionCode { distance: u32 },
    LdpcCode { n: u32, k: u32 },
}

pub struct QecAnalysis {
    pub code: QecCode,
    pub logical_error_rate: f64,
    pub physical_error_rate: f64,
    pub overhead_qubits: u32,   // physical / logical
    pub syndrome_depth: u32,
}

impl QecAnalysis {
    pub fn analyse(circuit: &Circuit, code: QecCode, phys_error_rate: f64) -> Self { ... }
}
```

**File:** `crates/lift-quantum/src/topology.rs`

```rust
// Add hardware topologies
pub enum Topology {
    Linear { n: usize },
    Grid { rows: usize, cols: usize },
    HeavyHex { n: usize },           // IBM heavy-hex lattice
    AllToAll { n: usize },           // Trapped ions
    Tree { n: usize },
    Custom { adjacency: Vec<Vec<usize>> },
}

pub struct HardwareTopology {
    pub topology: Topology,
    pub calibration: Vec<GateCalibration>,
    pub provider: Provider,
}

pub enum Provider {
    IbmKyoto,
    IbmEagle,
    Rigetti,
    IonQ,
    Quantinuum,
    Simulator,
}
```

---

### 2.3 `lift-hybrid` — Deepen Hybrid Capabilities

**File:** `crates/lift-hybrid/src/ops.rs`

```rust
// New hybrid operations
pub enum HybridOp {
    // Existing ops...

    // VQC Layer (Variational Quantum Circuit as a differentiable layer)
    VqcLayer {
        num_qubits: u32,
        num_layers: u32,
        encoding: EncodingStrategy,
        ansatz: AnsatzType,
    },

    // Quantum Kernel (inner product in feature space)
    QuantumKernel {
        num_qubits: u32,
        feature_map: FeatureMap,
    },

    // QAOA Layer
    QaoaLayer {
        num_qubits: u32,
        p: u32,         // number of rounds
        problem: QaoaProblem,
    },

    // VQE Component
    VqeAnsatz {
        num_qubits: u32,
        num_layers: u32,
        ansatz: AnsatzType,
    },

    // Co-execution with explicit scheduling
    CoExecute {
        classical_block: RegionId,
        quantum_block: RegionId,
        sync_policy: SyncPolicy,
    },

    // Data transfer
    GpuToQpu { dtype: DataType, encoding: EncodingStrategy },
    QpuToGpu { measurement_basis: MeasurementBasis },
}

pub enum AnsatzType {
    HardwareEfficient,
    StronglyEntangling,
    TwoLocal { entanglement: EntanglementPattern },
    Custom { circuit: CircuitTemplate },
}

pub enum SyncPolicy {
    Blocking,
    Asynchronous { timeout_ms: u32 },
    Pipeline { buffer_size: u32 },
}
```

**File:** `crates/lift-hybrid/src/gradient.rs`

```rust
// Complete gradient engine
pub struct GradientEngine {
    pub method: GradientMethod,
    pub shots: u32,
    pub parallel_evaluations: u32,
}

pub enum GradientMethod {
    ParameterShift,
    FiniteDifference { epsilon: f64 },
    Spsa { a: f64, c: f64 },
    AdjointDifferentiation,   // Exact, simulation only
    StochasticParameterShift { num_samples: u32 },
}

impl GradientEngine {
    // Compute gradient with checkpointing to reduce circuit evaluations
    pub fn compute_with_checkpoint(
        &self,
        circuit: &Circuit,
        params: &[f64],
        checkpoint_every: u32,
    ) -> Vec<f64> { ... }

    // Joint gradient: classical AD + quantum parameter shift
    pub fn joint_gradient(
        &self,
        classical_graph: &Region,
        quantum_circuit: &Circuit,
        params: &[f64],
    ) -> (Vec<f64>, Vec<f64>) { ... }
}
```

---

## 3. Phase 2 — Precision & Prediction (Months 2–5)

### 3.1 `lift-sim` — Make Simulation a Decision Engine

**File:** `crates/lift-sim/src/analysis.rs`

```rust
// Upgrade: track symbolic dimensions
pub struct SymbolicShape {
    pub dims: Vec<SymbolicDim>,
}

pub enum SymbolicDim {
    Static(u64),
    Dynamic,
    Symbolic(String),    // e.g. "batch", "seq_len"
}

// Exact FLOPs with symbolic support
pub struct FlopAnalysis {
    pub static_flops: Option<u128>,     // exact if all dims known
    pub symbolic_flops: Option<String>, // e.g. "2 * B * S * H * D"
    pub flops_per_element: u64,         // for dynamic shapes
}
```

**File:** `crates/lift-sim/src/quantum_sim.rs`

```rust
// Upgrade noise model: use Kraus operators
pub struct QuantumSimulator {
    pub backend: SimulatorBackend,
    pub noise_model: NoiseModel,
}

pub enum SimulatorBackend {
    StateVector { max_qubits: u32 },
    DensityMatrix { max_qubits: u32 },
    MatrixProductState { max_bond_dim: u32 },
    Stabilizer,   // Clifford circuits only, exponentially faster
    Pauli,        // Pauli noise approximation
}

pub struct NoiseModel {
    pub gate_errors: HashMap<String, KrausChannel>,
    pub readout_errors: Vec<f64>,
    pub t1_us: Vec<f64>,
    pub t2_us: Vec<f64>,
    pub crosstalk: HashMap<(usize, usize), f64>,
    pub source: NoiseModelSource,
}

pub enum NoiseModelSource {
    Default,
    Hardware(HardwareCalibration),
    File(PathBuf),
    Learned(GnnNoisePredictor),
}

// Fidelity propagation using channel composition
impl QuantumSimulator {
    pub fn estimate_fidelity_kraus(&self, circuit: &Circuit) -> f64 {
        // compose all gate channels
        // apply decoherence based on T1/T2 and circuit time
        // return average circuit fidelity
    }
}
```

**File:** `crates/lift-sim/src/budget.rs` (upgrade existing)

```rust
// Reactive budget system: propose fixes when violated
pub struct BudgetResult {
    pub passed: bool,
    pub violations: Vec<BudgetViolation>,
    pub suggestions: Vec<OptimisationSuggestion>,
}

pub struct BudgetViolation {
    pub constraint: BudgetConstraint,
    pub actual: f64,
    pub limit: f64,
    pub excess_pct: f64,
}

pub enum OptimisationSuggestion {
    ReducePrecision { from: DataType, to: DataType, savings_pct: f64 },
    EnableFlashAttention { latency_reduction_pct: f64 },
    EnableQuantisation { dtype: DataType, memory_reduction_pct: f64 },
    ReduceSequenceLength { from: u32, to: u32 },
    EnableGradientCheckpointing { memory_saving_pct: f64 },
    GateCancellation { gates_removed: u32, fidelity_gain: f64 },
    CustomPass { pass_name: String, estimated_benefit: String },
}
```

### 3.2 `lift-predict` — Add GNN Predictor

**File:** `crates/lift-predict/src/gnn.rs` (new file)

```rust
// GNN Performance Predictor (inference only — model loaded from file)
pub struct GnnPredictor {
    pub model_path: PathBuf,
    pub hardware_target: HardwareTarget,
    pub confidence_threshold: f64,
}

// Graph representation for the IR
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    pub edges: Vec<IrEdge>,
    pub hardware_features: HardwareFeatures,
}

pub struct IrNode {
    pub op_type: String,
    pub flops: u128,
    pub memory_bytes: u64,
    pub shape_features: Vec<f32>,   // encoded input/output shapes
    pub dtype_features: Vec<f32>,   // one-hot encoded dtype
}

pub struct HardwareFeatures {
    pub peak_flops_fp16: f64,
    pub memory_bandwidth: f64,
    pub l2_cache_mb: f64,
    pub sm_count: u32,
}

pub struct PredictionResult {
    pub latency_ms: f64,
    pub memory_gb: f64,
    pub confidence: f64,
    pub method: PredictionMethod,
}

pub enum PredictionMethod {
    Gnn,
    RooflineFallback,   // used when confidence < threshold
    Analytical,
}

impl GnnPredictor {
    pub fn predict(&self, ir: &IrGraph) -> PredictionResult {
        if self.model_available() {
            let pred = self.run_inference(ir);
            if pred.confidence >= self.confidence_threshold {
                return pred;
            }
        }
        // Fallback to analytical roofline
        self.roofline_predict(ir)
    }
}
```

**File:** `crates/lift-predict/src/energy.rs` (new file)

```rust
// Energy and carbon estimation
pub struct EnergyModel {
    pub hardware: HardwareTarget,
    pub region: GridRegion,
}

pub enum GridRegion {
    UsEast,
    UsWest,
    EuWest,
    AsiaPacific,
    Custom { co2_grams_per_kwh: f64 },
}

pub struct EnergyEstimate {
    pub compute_kwh: f64,
    pub memory_kwh: f64,
    pub total_kwh: f64,
    pub co2_grams: f64,
}

impl EnergyModel {
    pub fn estimate(&self, latency_ms: f64, utilisation: f64) -> EnergyEstimate {
        // TDP * latency * utilisation => energy
        // energy * grid_intensity => CO2
    }
}
```

---

## 4. Phase 3 — Developer Experience (Months 3–6)

### 4.1 `lift-ast` — Language Improvements

**File:** `crates/lift-ast/src/parser.rs` — Add shorthand syntax

```
// CURRENT: verbose transformer block
%q = "tensor.linear"(%x, %W_q, %b_q) : (tensor<B x S x H xf32>, ...) -> ...
%k = "tensor.linear"(%x, %W_k, %b_k) : ...
%v = "tensor.linear"(%x, %W_v, %b_v) : ...
%out = "tensor.attention"(%q, %k, %v) : ...

// NEW: transformer block shorthand
@transformer_block %x with heads=8, hidden=512, ffn_dim=2048 {
    attn: multi_head_attention,
    norm: layernorm,
    ffn: [linear, gelu, linear],
}
```

**New keywords to add to lexer:**
```
"@transformer_block"
"@residual"
"@vqe_layer"
"@qaoa_layer"
"@encoder"
"@decoder"
let, in, where
```

**File:** `crates/lift-ast/src/formatter.rs` (new file)

```rust
// lift fmt — canonical formatter
pub struct LiftFormatter {
    pub indent: u32,
    pub max_line_length: u32,
    pub sort_attributes: bool,
}

impl LiftFormatter {
    pub fn format(&self, source: &str) -> Result<String, FormatError> { ... }
}
```

### 4.2 `lift-config` — .lith Improvements

**File:** `crates/lift-config/src/inheritance.rs` (new file)

```toml
# Example: .lith inheritance
extends = "base_gpu_h100.lith"

# Override only what changes
[budget]
max_latency_ms = 50    # stricter than base
```

```rust
// Config inheritance system
pub struct LithConfig {
    pub extends: Option<PathBuf>,
    // ... existing fields ...
}

impl LithConfig {
    pub fn resolve(path: &PathBuf) -> Result<ResolvedConfig, ConfigError> {
        // load base, apply overrides, validate
    }

    pub fn validate(&self) -> Vec<ConfigWarning> {
        // type-check all fields
        // warn on conflicting settings
        // suggest missing required fields
    }
}
```

**Predefined base configs to ship:**

```
lift/configs/
├── base_gpu_a100.lith
├── base_gpu_h100.lith
├── base_cpu_llvm.lith
├── base_ibm_kyoto.lith
├── base_rigetti.lith
└── base_simulator.lith
```

### 4.3 `lift-python` — Python Bindings

**File:** `crates/lift-python/src/lib.rs`

```rust
// Full PyO3 bindings
use pyo3::prelude::*;

#[pymodule]
fn lift(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyContext>()?;
    m.add_class::<PyModule>()?;
    m.add_class::<PyCompiler>()?;
    m.add_function(wrap_pyfunction!(from_torch, m)?)?;
    m.add_function(wrap_pyfunction!(from_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(from_qiskit, m)?)?;
    Ok(())
}

#[pyfunction]
fn from_torch(model: &PyAny, example_inputs: &PyAny) -> PyResult<PyModule> {
    // Use PyTorch FX to trace, then import into LIFT IR
}

#[pyfunction]
fn from_qiskit(circuit: &PyAny) -> PyResult<PyModule> {
    // Convert QuantumCircuit → LIFT quantum IR
}
```

---

## 5. Phase 4 — Hardware Backends (Months 4–8)

### 5.1 `lift-export` — CUDA Backend

**File:** `crates/lift-export/src/cuda.rs` (new file)

```rust
pub struct CudaBackend {
    pub target_arch: CudaArch,
    pub use_tensor_cores: bool,
    pub use_cublas: bool,
    pub use_cudnn: bool,
    pub use_cutlass: bool,
}

pub enum CudaArch {
    Sm80,   // A100
    Sm90,   // H100
    Sm86,   // RTX 3090
}

impl CudaBackend {
    // Priority order for code generation:
    // 1. cuBLAS call (for matmul, batched matmul)
    // 2. cuDNN call (for conv, pooling, batch norm)
    // 3. CUTLASS template (for fused ops)
    // 4. Custom PTX kernel (fallback)

    pub fn emit_matmul(&self, op: &MatmulOp) -> String {
        // emit cublasGemmEx or custom Tensor Core kernel
    }

    pub fn emit_flash_attention(&self, op: &AttentionOp) -> String {
        // emit FlashAttention-2 kernel call
    }

    pub fn emit_fused_kernel(&self, ops: &[TensorOp]) -> String {
        // fuse into single CUDA kernel to reduce memory bandwidth
    }
}
```

**File:** `crates/lift-export/src/qasm3.rs` (upgrade existing)

```rust
// Upgrade to full OpenQASM 3.0 compliance
pub struct Qasm3Exporter {
    pub hardware: Option<HardwareTarget>,  // for gate decomposition
    pub optimize_for_hardware: bool,
}

impl Qasm3Exporter {
    // Decompose gates to hardware-native basis
    pub fn decompose_to_basis(
        &self,
        gate: &QuantumGate,
        basis: &[String],
    ) -> Vec<QuantumGate> {
        // e.g. IBM: {RZ, SX, X, CX}
        // e.g. Rigetti: {RZ, RX, CZ}
        // e.g. IonQ: {GPI, GPI2, MS}
    }

    // Generate pulse schedule for IBM backends
    pub fn emit_pulse_schedule(&self, circuit: &Circuit) -> String { ... }
}
```

### 5.2 `lift-import` — Complete All Importers

**File:** `crates/lift-import/src/onnx.rs` (complete to 100%)

```rust
// Support ONNX opset 17+
// Missing ops to add: Einsum, LSTM, GRU, ScatterElements,
//                     NonMaxSuppression, RoiAlign, etc.
```

**File:** `crates/lift-import/src/qiskit.rs` (new file)

```rust
pub struct QiskitImporter;

impl QiskitImporter {
    // Import QuantumCircuit (via JSON or QASM string)
    pub fn from_qasm3(qasm: &str) -> Result<LiftModule, ImportError> { ... }

    // Import with noise model (from IBM backend.properties())
    pub fn from_qasm3_with_noise(
        qasm: &str,
        backend_properties: &str,
    ) -> Result<(LiftModule, NoiseModel), ImportError> { ... }
}
```

**File:** `crates/lift-import/src/tensorflow.rs` (new file)

```rust
pub struct TensorflowImporter;

impl TensorflowImporter {
    pub fn from_saved_model(path: &Path) -> Result<LiftModule, ImportError> { ... }
    pub fn from_keras_json(json: &str) -> Result<LiftModule, ImportError> { ... }
}
```

---

## 6. Phase 5 — Stability & Production (Months 6–12)

### 6.1 `lift-opt` — More Optimisation Passes

**File:** `crates/lift-opt/src/passes/` (add these files)

```
lift-opt/src/passes/
├── dce.rs                   ✅ exists
├── constant_fold.rs         ✅ exists
├── tensor_fusion.rs         ✅ exists (extend patterns)
├── gate_cancel.rs           ✅ exists
├── canonicalize.rs          ✅ exists
├── flash_attention.rs       🆕 Replace attention with FlashAttention when seq_len > 512
├── kv_cache.rs              🆕 Pre-allocate KV cache for LLM inference
├── quantisation.rs          🆕 INT8/FP8 dynamic and static quantisation
├── rotation_merge.rs        🆕 Rz(a).Rz(b) -> Rz(a+b) on same qubit
├── layout_mapping.rs        🆕 SABRE routing for QPU physical layout
├── zne_mitigation.rs        🆕 Zero Noise Extrapolation pass
├── common_subexpr.rs        🆕 Common subexpression elimination
├── loop_invariant.rs        🆕 Hoist loop-invariant operations
├── memory_layout.rs         🆕 NCHW <-> NHWC conversion
└── noise_aware_schedule.rs  🆕 Reorder gates to minimise decoherence
```

**Each pass must implement:**

```rust
pub trait OptPass {
    fn name(&self) -> &'static str;
    fn run(&self, ctx: &mut Context) -> PassResult;
    fn is_semantics_preserving(&self) -> bool;   // always true
    fn applicable_to(&self, op: &Operation) -> bool;
}
```

**FlashAttention pass specification:**

```rust
// TRIGGER: op is tensor.attention AND target is GPU AND seq_len > 512
// ACTION:  replace with tensor.flash_attention (same semantics, tiled execution)
// GAIN:    O(n^2) memory -> O(n), 10-20x speedup on long sequences
```

**Gate rotation merging specification:**

```rust
// TRIGGER: two Rz gates on the same qubit with no other gates between them
// ACTION:  Rz(a).Rz(b) -> Rz(a+b)
// EXTEND:  Rx(a).Rx(b) -> Rx(a+b), Ry(a).Ry(b) -> Ry(a+b)
// Also:    Rz(0) -> identity (remove), Rz(2π) -> identity (remove)
```

### 6.2 `lift-tests` — Expand Test Suite

**Target: 1000+ tests, 100% pass rate**

**New test files to create:**

```
crates/lift-tests/src/
├── test_core_comprehensive.rs         ✅ 25 tests
├── test_ast_comprehensive.rs          ✅ 24 tests
├── test_tensor_comprehensive.rs       ✅ 21 tests — extend to 50+
├── test_quantum_comprehensive.rs      ✅ 20 tests — extend to 50+
├── test_opt_comprehensive.rs          ✅ 15 tests — extend to 40+
├── test_integration_pipeline.rs       ✅ 13 tests — extend to 30+
├── test_benchmarks.rs                 ✅ 13 tests — extend to 30+
├── test_attention_variants.rs         🆕 All attention op variants
├── test_conv_variants.rs              🆕 Conv1d/2d/3d/transposed/dilated
├── test_recurrent.rs                  🆕 LSTM, GRU, RNN cells
├── test_advanced_math.rs              🆕 Einsum, FFT, SVD
├── test_quantisation.rs               🆕 INT4, INT8, FP8 quantisation
├── test_kraus_noise.rs                🆕 Kraus channel composition
├── test_qec.rs                        🆕 QEC code analysis
├── test_topology_extended.rs          🆕 Heavy-hex, all-to-all, custom
├── test_gradient_engine.rs            🆕 Parameter shift, SPSA, finite diff
├── test_flash_attention_pass.rs       🆕 Pass trigger, correctness
├── test_rotation_merge_pass.rs        🆕 Rz(a).Rz(b) -> Rz(a+b)
├── test_layout_mapping_pass.rs        🆕 SABRE routing correctness
├── test_budget_reactive.rs            🆕 Suggestions when budget violated
├── test_config_inheritance.rs         🆕 .lith inheritance and validation
├── test_energy_model.rs               🆕 Energy/CO2 estimation
├── test_python_bindings.rs            🆕 PyO3 round-trip
├── test_differential.rs               🆕 LIFT vs PyTorch output comparison
└── test_fuzzing.rs                    🆕 Property-based tests (proptest)
```

**Property-based tests (using `proptest` crate):**

```rust
proptest! {
    #[test]
    fn prop_flops_always_positive(shape in any_valid_shape()) {
        let flops = matmul_flops(&shape);
        prop_assert!(flops > 0);
    }

    #[test]
    fn prop_fidelity_in_0_1(n_gates in 0u32..1000) {
        let fidelity = circuit_fidelity(n_gates, 0, 0.999, 0.99);
        prop_assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }

    #[test]
    fn prop_dce_preserves_semantics(prog in any_valid_program()) {
        let original = interpret(&prog);
        let optimised = interpret(&dce_pass(&prog));
        prop_assert_eq!(original, optimised);
    }
}
```

### 6.3 Formal Verification for Critical Passes

**File:** `crates/lift-opt/src/verify/` (new directory)

```rust
// SMT-based semantic equivalence checker (using z3 bindings)
pub struct SemanticChecker {
    pub ctx: z3::Context,
}

impl SemanticChecker {
    // Check: original_ir ≡ optimised_ir for all inputs
    pub fn check_equivalence(
        &self,
        original: &Region,
        optimised: &Region,
    ) -> EquivalenceResult { ... }
}

pub enum EquivalenceResult {
    Equivalent,
    NotEquivalent { counterexample: InputValues },
    Timeout,
    Unknown,
}
```

**Only needed for the most complex passes:**
- `tensor_fusion` (MatMul + Bias + ReLU pattern)
- `gate_cancel` (quantum self-inverse pairs)
- `rotation_merge` (Rz(a).Rz(b) = Rz(a+b))

---

## 7. Cross-Cutting Concerns

### 7.1 Error Handling — Upgrade All Crates

```rust
// Use thiserror for structured errors in every crate
// Pattern: one LiftError enum per crate

#[derive(Debug, thiserror::Error)]
pub enum LiftError {
    #[error("Type mismatch at {location}: expected {expected}, found {found}")]
    TypeMismatch { location: SourceLocation, expected: String, found: String },

    #[error("Qubit {qubit} used more than once (linear type violation) at {location}")]
    QubitReuse { qubit: String, location: SourceLocation },

    #[error("Budget violated: {constraint} = {actual:.2} exceeds limit {limit:.2}")]
    BudgetViolation { constraint: String, actual: f64, limit: f64 },

    #[error("Unknown operation '{name}' in dialect '{dialect}'")]
    UnknownOperation { name: String, dialect: String },
}

pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}
```

### 7.2 Logging — Structured Logs

```rust
// Add to Cargo.toml of each crate:
// tracing = "0.1"
// tracing-subscriber = { version = "0.3", features = ["json"] }

// Usage in each pass:
tracing::info!(
    pass = "tensor_fusion",
    ops_fused = 3,
    memory_savings_pct = 35.2,
    "Tensor fusion applied"
);

tracing::debug!(
    gate = "Rz",
    theta_before = a,
    theta_after = a + b,
    "Rotation merged"
);
```

### 7.3 Benchmarks — Criterion Integration

```rust
// crates/lift-tests/benches/compilation_speed.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_llama7b_parse(c: &mut Criterion) {
    c.bench_function("llama7b_parse", |b| {
        b.iter(|| parse_llama7b_lif())
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    // Parse + Verify + Analyse + Optimise must complete in < 1s for LLaMA-7B
    c.bench_function("full_pipeline_llama7b", |b| {
        b.iter(|| run_full_pipeline("examples/llama7b.lif"))
    });
}

criterion_group!(benches, bench_llama7b_parse, bench_full_pipeline);
criterion_main!(benches);
```

### 7.4 Cargo.toml Additions

```toml
# Root Cargo.toml — add to workspace dependencies
[workspace.dependencies]
thiserror   = "1.0"
tracing     = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
proptest    = "1.0"
criterion   = { version = "0.5", features = ["html_reports"] }
rayon       = "1.8"
dashmap     = "5"
serde       = { version = "1.0", features = ["derive"] }
serde_json  = "1.0"
chrono      = { version = "0.4", features = ["serde"] }
ndarray     = "0.15"

# For Python bindings crate
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }

# For SMT verification
z3 = "0.12"

# For GNN inference
ort = "1.16"    # ONNX Runtime bindings for loading GNN model
```

---

## 8. File-by-File Modification Map

This is the exact list of files Windsurf should create or modify.

### New Files to Create

```
crates/lift-tensor/src/flops.rs           # Exact FLOPs for all ops
crates/lift-tensor/src/attention.rs       # Attention variants
crates/lift-tensor/src/conv.rs            # All conv variants
crates/lift-tensor/src/recurrent.rs       # LSTM, GRU, RNN
crates/lift-tensor/src/advanced_math.rs   # Einsum, FFT, SVD
crates/lift-quantum/src/kraus.rs          # Kraus channel model
crates/lift-quantum/src/qec.rs            # QEC analysis
crates/lift-quantum/src/calibration.rs    # Hardware calibration data
crates/lift-hybrid/src/gradient.rs        # Full gradient engine
crates/lift-hybrid/src/vqc.rs             # VQC layer
crates/lift-hybrid/src/qaoa.rs            # QAOA layer
crates/lift-sim/src/budget.rs             # Reactive budget (upgrade)
crates/lift-sim/src/hybrid_sim.rs         # Co-simulation GPU+QPU
crates/lift-predict/src/gnn.rs            # GNN predictor
crates/lift-predict/src/energy.rs         # Energy/CO2 model
crates/lift-opt/src/passes/flash_attention.rs
crates/lift-opt/src/passes/kv_cache.rs
crates/lift-opt/src/passes/quantisation.rs
crates/lift-opt/src/passes/rotation_merge.rs
crates/lift-opt/src/passes/layout_mapping.rs
crates/lift-opt/src/passes/zne_mitigation.rs
crates/lift-opt/src/passes/common_subexpr.rs
crates/lift-opt/src/passes/noise_aware_schedule.rs
crates/lift-opt/src/verify/mod.rs         # SMT equivalence checker
crates/lift-import/src/qiskit.rs
crates/lift-import/src/tensorflow.rs
crates/lift-import/src/cirq.rs
crates/lift-export/src/cuda.rs
crates/lift-export/src/rocm.rs
crates/lift-export/src/xla.rs
crates/lift-config/src/inheritance.rs
crates/lift-config/src/validation.rs
crates/lift-python/src/lib.rs             # Full PyO3 bindings
crates/lift-ast/src/formatter.rs          # lift fmt
configs/base_gpu_a100.lith
configs/base_gpu_h100.lith
configs/base_ibm_kyoto.lith
configs/base_rigetti.lith
examples/llama7b.lif
examples/vqe_h2.lif
examples/qnn_mnist.lif
examples/attention_transformer.lif
```

### Files to Modify (extend, do not replace)

```
crates/lift-tensor/src/ops.rs         # Add 40+ new ops
crates/lift-tensor/src/types.rs       # Add annotated types
crates/lift-quantum/src/gates.rs      # Add 20+ new gates
crates/lift-quantum/src/noise.rs      # Upgrade to Kraus model
crates/lift-quantum/src/topology.rs   # Add HeavyHex, AllToAll
crates/lift-hybrid/src/ops.rs         # Add VQC, QAOA, CoExecute
crates/lift-sim/src/analysis.rs       # Add symbolic shapes
crates/lift-sim/src/quantum_sim.rs    # Upgrade to density matrix
crates/lift-opt/src/passes/tensor_fusion.rs  # More patterns
crates/lift-opt/src/passes/gate_cancel.rs    # More pairs, rotation angles
crates/lift-import/src/onnx.rs        # Complete to opset 17+
crates/lift-import/src/pytorch.rs     # Complete to 100%
crates/lift-import/src/qasm.rs        # Complete QASM 3.0
crates/lift-export/src/llvm.rs        # Fix remaining 30%
crates/lift-export/src/qasm_export.rs # Fix remaining 60%
crates/lift-config/src/parser.rs      # Add inheritance
crates/lift-cli/src/main.rs           # Add compile, simulate, predict
crates/lift-tests/src/test_*.rs       # Add tests to all files
```

---

## 9. Test Strategy

### 9.1 Test Naming Convention

```rust
// Format: test_{component}_{scenario}_{expected}
fn test_flash_attention_seq1024_triggers_pass() { ... }
fn test_rotation_merge_rz_a_rz_b_produces_rz_sum() { ... }
fn test_budget_violated_flops_suggests_quantisation() { ... }
fn test_kraus_compose_two_depolarizing_channels() { ... }
```

### 9.2 Test Categories

| Category | Count Target | Description |
|----------|-------------|-------------|
| Unit | 400+ | Each op, gate, pass in isolation |
| Integration | 100+ | Full pipeline parse→export |
| Differential | 50+ | LIFT output == PyTorch/Qiskit output |
| Benchmark | 50+ | Known models with validated numbers |
| Stress | 30+ | Large circuits, many ops, deep nesting |
| Edge cases | 50+ | Empty, zero, single-element, max dims |
| Property-based | 100+ | proptest invariants |
| Regression | 200+ | Previously-found bugs, never re-broken |

### 9.3 CI Pipeline

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    steps:
      - run: cargo test --workspace         # all 165+ tests
      - run: cargo clippy -- -D warnings    # no warnings
      - run: cargo audit                    # no vulnerabilities

  benchmarks:
    steps:
      - run: cargo bench --bench compilation_speed
      - run: python scripts/compare_bench.py  # fail if >5% regression

  coverage:
    steps:
      - run: cargo tarpaulin --out Xml
      - run: python scripts/check_coverage.py 85  # fail if <85%

  differential:
    steps:
      - run: python tests/differential/run_pytorch.py
      - run: python tests/differential/run_qiskit.py
```

---

## 10. Priority Matrix

Use this to decide what to implement first.

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| Attention variants (MQA, GQA, Flash) | Critical | Medium | **P0** |
| Kraus noise model | Critical | Medium | **P0** |
| FlashAttention pass | Critical | High | **P0** |
| Rotation merge pass | High | Low | **P0** |
| Budget reactive suggestions | High | Medium | **P1** |
| CUDA backend (matmul + attention) | Critical | Very High | **P1** |
| Conv variants (all types) | High | Medium | **P1** |
| LSTM/GRU/RNN | High | Medium | **P1** |
| Python bindings (PyO3) | High | High | **P1** |
| .lith inheritance | Medium | Low | **P1** |
| Qiskit importer (complete) | High | Medium | **P2** |
| GNN predictor (inference) | Medium | High | **P2** |
| Einsum, FFT, SVD | Medium | Medium | **P2** |
| QEC analysis | Medium | High | **P2** |
| Energy/CO2 model | Medium | Low | **P2** |
| SABRE layout mapping | Medium | High | **P2** |
| ZNE mitigation pass | Medium | High | **P3** |
| TensorFlow importer | Low | High | **P3** |
| XLA export | Low | High | **P3** |
| SMT formal verification | Low | Very High | **P3** |
| GNN predictor (training) | Medium | Very High | **P3** |

---

## Appendix A — Version Targets

| Version | Milestone | Key Deliverables |
|---------|-----------|-----------------|
| **v0.3.0** | Month 3 | All attention variants, Kraus noise, 5 new passes, 300+ tests |
| **v0.4.0** | Month 5 | CUDA backend (matmul+attention), GNN predictor, Python bindings |
| **v0.5.0** | Month 7 | Full quantum gate set, QEC, layout mapping, ZNE |
| **v0.6.0** | Month 9 | All importers complete, hybrid co-simulation, energy model |
| **v1.0.0** | Month 12 | All backends, 1000+ tests, published benchmarks, arXiv preprint |

---

## Appendix B — Backward Compatibility Rules

1. **Never remove** an existing `TensorOp` variant — add deprecation attribute instead.
2. **Never change** the SSA definition invariant.
3. **Never change** the qubit linearity rule.
4. **Never rename** existing CLI commands — add aliases if needed.
5. **Never break** the `.lif` parser for valid existing files.
6. **Never change** flop_count() semantics for existing ops.
7. All new ops must have `flop_count()` and `memory_bytes()` implemented.
8. All new passes must produce output that passes `lift verify`.

---

*End of LIFT Improvement Plan — v0.2.0 → v1.0.0*
*Generated from: Strategy, Precision, Coverage, and Stability research documents.*
*Ready for Windsurf automated implementation.*