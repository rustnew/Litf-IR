# LIFT — Comprehensive Manual of Problems and Solutions

> **LIFT** (Layered Intermediate Framework for Tensors & Qubits) is a unified compiler
> that transforms high-level descriptions into optimised executable code
> for GPUs, CPUs, and quantum processors.

---

## Table of Contents

- [Part I — LIFT Architecture](#part-i--lift-architecture)
- [Part II — Classical AI Problems](#part-ii--classical-ai-problems)
- [Part III — Quantum Problems](#part-iii--quantum-problems)
- [Part IV — Hybrid AI + Quantum Problems](#part-iv--hybrid-ai--quantum-problems)
- [Part V — Cross-Cutting Capabilities](#part-v--cross-cutting-capabilities)
- [Part VI — Quick Reference](#part-vi--quick-reference)

---

# Part I — LIFT Architecture

## 1.1 The 12 Framework Crates

| Crate | Role |
|-------|------|
| **lift-core** | SSA IR, types, verifier, printer, dialect registry |
| **lift-ast** | Lexer, parser, IR builder from `.lif` files |
| **lift-tensor** | 90+ AI/ML operations, shape inference, FLOPs |
| **lift-quantum** | 50+ gates, noise, topology, QEC, Kraus channels |
| **lift-hybrid** | Encoding, gradients, VQC/VQE/QAOA, GPU-QPU transfer |
| **lift-opt** | 11 optimisation passes |
| **lift-sim** | Static analysis, GPU/quantum cost models, energy, budgets |
| **lift-predict** | GPU roofline prediction and quantum fidelity/shots |
| **lift-export** | Export to LLVM IR and OpenQASM 3.0 |
| **lift-config** | `.lith` configuration file parsing |
| **lift-import** | Import from external formats |
| **lift-cli** | Command-line interface |

## 1.2 The 6-Stage Pipeline

```
.lif → [1. PARSE & VERIFY] → [2. ANALYSE] → [3. OPTIMISE] → [4. PREDICT] → [5. EXPORT] → [6. FEEDBACK]
```

1. **Parse & Verify**: Lexer → Parser → SSA → type checking + qubit linearity
2. **Analyse**: FLOPs, memory, circuit depth, estimated fidelity
3. **Optimise**: 11 classical + quantum + hybrid passes
4. **Predict**: GPU roofline + QPU noise model
5. **Export**: LLVM IR (GPU/CPU) + OpenQASM 3.0 (QPU)
6. **Feedback**: Budget, energy, CO2, predicted vs. actual comparison

## 1.3 The Three Dialects

| Dialect | Prefix | Domain | Example |
|---------|--------|--------|---------|
| **tensor** | `tensor.` | Classical AI / ML | `tensor.conv2d`, `tensor.attention` |
| **quantum** | `quantum.` | Quantum computing | `quantum.h`, `quantum.cx` |
| **hybrid** | `hybrid.` | Classical-quantum bridge | `hybrid.encode`, `hybrid.vqc_layer` |

---

# Part II — Classical AI Problems

## 2.1 Computer Vision (CNN)

### Problem

Classify, detect, or segment images: medical X-rays, satellite photos, industrial quality control, autonomous vehicles.

### Solution LIFT

```
func @image_classifier(%img: tensor<1x3x224x224xf32>,
                        %w1: tensor<64x3x7x7xf32>,
                        %w2: tensor<128x64x3x3xf32>,
                        %wfc: tensor<128x10xf32>,
                        %b: tensor<10xf32>) -> tensor<1x10xf32> {
  %c1 = tensor.conv2d %img, %w1
  %r1 = tensor.relu %c1
  %p1 = tensor.maxpool2d %r1
  %c2 = tensor.conv2d %p1, %w2
  %r2 = tensor.relu %c2
  %g  = tensor.global_avgpool %r2
  %fc = tensor.matmul %g, %wfc
  %out = tensor.add %fc, %b
  return %out
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Parse** | Builds the SSA graph, verifies that `conv2d` receives compatible 4D tensors |
| **Analyse** | Computes exact FLOPs (`2 * N * C * H * W * K * K * F`), VRAM, peak memory |
| **Optimise** | **Tensor fusion**: `conv2d + batchnorm + relu → fused_conv_bn_relu` (-30% memory). **DCE**: removes unused weights |
| **Predict** | Roofline A100/H100: determines whether the CNN is compute-bound or memory-bound |
| **Export** | LLVM IR with cuDNN calls (convolutions) and cuBLAS calls (matmuls) |
| **Budget** | Verifies that VRAM does not exceed 80 GB (A100) |

### Operations Used

`Conv2D`, `Conv1D`, `Conv3D`, `DepthwiseConv2D`, `DilatedConv2D`, `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D`, `AdaptiveAvgPool2D`, `GlobalAvgPool`, `ReLU`, `BatchNorm`, `MatMul`, `Linear`, `Softmax`.

### Variants

- **Classification**: CNN → softmax → label
- **Object detection**: backbone → region proposals → bounding boxes
- **Segmentation**: U-Net (`UNetDownBlock`, `UNetUpBlock`)
- **Super-resolution**: `ConvTranspose2D` for upsampling

---

## 2.2 Natural Language Processing (Transformers)

### Problem

Translate, summarise, generate, or understand text: chatbots, search engines, sentiment analysis, code generation.

### Solution LIFT

```
func @transformer_block(%x: tensor<1x512x768xf32>,
                         %wq: tensor<768x768xf32>,
                         %wk: tensor<768x768xf32>,
                         %wv: tensor<768x768xf32>) -> tensor<1x512x768xf32> {
  %q = tensor.matmul %x, %wq
  %k = tensor.matmul %x, %wk
  %v = tensor.matmul %x, %wv
  %attn = tensor.multi_head_attention %q, %k, %v
  %n1 = tensor.layernorm %attn
  %ff = tensor.linear %n1, %wff
  %act = tensor.gelu %ff
  %out = tensor.layernorm %act
  return %out
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Computes the O(n^2) cost of attention for the sequence length |
| **Optimise** | **FlashAttention**: replaces `attention` with `flash_attention` (O(n) memory). **CSE**: eliminates redundant computations. **FusedAttentionLayerNorm**: single kernel |
| **Predict** | LLM inference = memory-bound. Training = compute-bound. LIFT identifies the regime |
| **Export** | Fused kernels for attention |

### 8 Attention Variants

| Operation | Usage | Memory |
|-----------|-------|--------|
| `Attention` | Standard | O(n^2) |
| `MultiHeadAttention` | GPT, BERT | O(n^2) |
| `MultiQueryAttention` | PaLM, Falcon (fast inference) | O(n) per head |
| `GroupedQueryAttention` | LLaMA 2 (trade-off) | O(n * G/H) |
| `FlashAttention` | Training and inference | O(n) |
| `SlidingWindowAttention` | Mistral (long sequences) | O(n * w) |
| `CrossAttention` | Translation, multimodal | O(n * m) |
| `PagedAttention` | LLM serving (vLLM) | O(n) per page |

### Use Cases

- **Chatbot / LLM**: N transformer blocks + `PagedAttention` for serving
- **Translation**: Encoder (`MultiHeadAttention`) + Decoder (`CrossAttention`)
- **Sentiment analysis**: BERT + classification
- **Code generation**: GPT + `SlidingWindowAttention` for long context

---

## 2.3 Recommendation Systems

### Problem

Recommend products, movies, or music to users based on their history.

### Solution LIFT

```
func @recommender(%user_id: tensor<1xi64>,
                   %item_ids: tensor<100xi64>,
                   %user_emb: tensor<100000x64xf32>,
                   %item_emb: tensor<500000x64xf32>) -> tensor<100xf32> {
  %u = tensor.embedding %user_id, %user_emb
  %items = tensor.embedding %item_ids, %item_emb
  %cat = tensor.concat %u, %items
  %h = tensor.linear %cat, %w1
  %a = tensor.relu %h
  %scores = tensor.linear %a, %w2
  %probs = tensor.sigmoid %scores
  return %probs
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Embedding table size (100k * 64 * 4 = 25 MiB for users) |
| **Optimise** | **SparseEmbedding** for large tables. **FusedLinearReLU** for dense layers |
| **Predict** | Recommendation systems = memory-bound (large embeddings) |
| **Budget** | Verifies that tables fit in VRAM for real-time serving |

### Key Operations

`Embedding`, `SparseEmbedding`, `Linear`, `ReLU`, `Sigmoid`, `Concat`, `Gather`, `Scatter`, `TopK`.

---

## 2.4 Time Series (LSTM / GRU / RNN)

### Problem

Predict future values from past sequences: weather forecasting, stock prices, predictive maintenance, ECG.

### Solution LIFT

```
func @forecast(%seq: tensor<1x100x16xf32>,
               %h0: tensor<1x64xf32>,
               %c0: tensor<1x64xf32>) -> tensor<1x1xf32> {
  %h, %c = tensor.lstm_cell %seq, %h0, %c0, %w
  %pred = tensor.matmul %h, %wfc
  return %pred
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | FLOPs per time step = 4 LSTM gates * matmul. 100 steps = 100x the cost |
| **Optimise** | **Constant folding**: constant biases. **Canonicalize**: redundant expressions |
| **Predict** | Training = compute-bound. Inference = memory-bound (cached states) |

### Recurrent Cells

| Operation | Description |
|-----------|-------------|
| `LSTMCell` | Long Short-Term Memory (4 gates) |
| `GRUCell` | Gated Recurrent Unit (3 gates) |
| `RNNCell` | RNN vanilla (tanh) |

---

## 2.5 Graph Neural Networks (GNN)

### Problem

Process graph-structured data: molecules, social networks, transportation systems, proteins.

### Solution LIFT

```
func @gnn_classify(%nodes: tensor<100x16xf32>,
                    %adj: tensor<100x100xf32>) -> tensor<1x2xf32> {
  %msg = tensor.gnn_message_passing %nodes, %adj, %w
  %act = tensor.relu %msg
  %graph = tensor.gnn_global_pooling %act
  %out = tensor.matmul %graph, %wfc
  %prob = tensor.softmax %out
  return %prob
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Message passing FLOPs: `N * N * D` (dense) or `E * D` (sparse) |
| **Optimise** | **SparseMatMul** if adjacency < 10% non-zero |
| **Predict** | GNN = memory-bound (irregular graph lookups) |

### GNN Aggregation Types

Sum, Mean, Max, Min (via `AggregationType`).

### Use Cases

- **Molecular properties**: solubility, toxicity
- **Drug discovery**: protein-ligand interaction
- **Social networks**: community detection
- **Transportation**: traffic prediction

---

## 2.6 Generative Models and Diffusion

### Problem

Generate images, audio, or video from noise or a text description: Stable Diffusion, DALL-E, voice synthesis.

### Solution LIFT

```
func @diffusion_step(%noisy: tensor<1x4x64x64xf32>,
                      %timestep: tensor<1xi64>,
                      %context: tensor<1x77x768xf32>) -> tensor<1x4x64x64xf32> {
  %t_emb = tensor.timestep_embedding %timestep
  %down = tensor.unet_down_block %noisy, %t_emb, %context
  %up = tensor.unet_up_block %down, %t_emb, %context
  return %up
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Total cost = N_denoising_steps * FLOPs_per_step (20-50 U-Net passes) |
| **Optimise** | **FlashAttention** for cross-attention blocks. **Tensor fusion** for residuals. **Checkpoint** to save memory |
| **Predict** | 1 SD 1.5 step on A100 ~ 50 ms. LIFT estimates and identifies attention as bottleneck |
| **Energy** | 50 steps * 50 ms = 2.5 s / image. LIFT computes consumption for 1M images/day |

### Key Operations

`UNetDownBlock`, `UNetUpBlock`, `TimestepEmbedding`, `CrossAttention`, `GroupNorm`, `SiLU`, `Conv2D`, `ConvTranspose2D`.

---

## 2.7 Scientific Computing (FFT, SVD, Linear Systems)

### Problem

Signal processing (FFT), matrix decomposition (SVD, eigenvalues), linear system solving (Ax = b).

### Solution LIFT

```
func @signal_analysis(%signal: tensor<1x8192xf32>) -> tensor<1x8192xf32> {
  %freq = tensor.fft %signal
  %filtered = tensor.mul %freq, %mask
  %result = tensor.ifft %filtered
  return %result
}

func @solve_system(%A: tensor<1000x1000xf32>,
                    %b: tensor<1000xf32>) -> tensor<1000xf32> {
  %x = tensor.solve %A, %b
  return %x
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | FFT: O(n log n). SVD: O(min(m,n)^2 * max(m,n)). Exact costs |
| **Optimise** | **Constant folding**: pre-computes constant masks |
| **Export** | LLVM IR with cuFFT calls (GPU) or FFTW (CPU) |

### Operations

`FFT`, `IFFT`, `SVD`, `Eig`, `Solve`, `Einsum`, `Cumsum`, `Sort`, `TopK`, `Where`, `Clamp`.

### Use Cases

- **Signal processing**: filtering, spectrogram, audio compression
- **Data analysis**: PCA via SVD, least squares
- **Physical simulation**: differential equations
- **Imaging**: Fourier reconstruction (MRI, CT scan)

---

## 2.8 Edge AI and Quantisation

### Problem

Deploy models on resource-constrained devices: smartphones, IoT, drones, embedded cameras.

### Solution LIFT

```
func @quantized_model(%img: tensor<1x3x224x224xf32>,
                       %w: tensor<64x3x3x3xf32>) -> tensor<1x10xf32> {
  %w_q = tensor.quantize_int4 %w        // weights in INT4 (8x less memory)
  %conv = tensor.conv2d %img, %w_q
  %out = tensor.dequantize_int4 %conv
  return %out
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Optimise** | **QuantisationPass**: annotates INT8/INT4/FP8-compatible ops, automatically inserts `quantize`/`dequantize` |
| **Analyse** | Recomputes FLOPs and memory after quantisation. INT4 divides memory by 8 |
| **Predict** | Estimates speedup on target hardware |
| **Budget** | Verifies the model fits on the device (e.g. 4 GB smartphone) |

### Quantisation Levels

| Operation | Precision | Memory Reduction | Usage |
|-----------|-----------|-----------------|-------|
| `Quantize` / `Dequantize` | INT8 | 4x | Server inference |
| `QuantizeInt4` / `DequantizeInt4` | INT4 | 8x | Smartphones, edge |
| `QuantizeFp8` / `DequantizeFp8` | FP8 (E4M3/E5M2) | 2x | H100 training |

---

## 2.9 Mixture of Experts (MoE)

### Problem

Build massively parameterised models (100B+) where only a fraction of parameters are activated per request: Mixtral, Switch Transformer.

### Solution LIFT

```
func @moe_layer(%x: tensor<1x512x768xf32>,
                 %gate: tensor<768x8xf32>) -> tensor<1x512x768xf32> {
  %tokens, %weights = tensor.moe_dispatch %x, %gate   // router: top-2 / 8 experts
  %expert_out = tensor.linear %tokens, %experts
  %combined = tensor.moe_combine %expert_out, %weights
  return %combined
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Only 2/8 experts active → effective FLOPs = 25% of total |
| **Optimise** | **Tensor fusion**: fuses dispatch + expert + combine if on same GPU |
| **Predict** | MoE = memory-bound (large parameters) despite moderate FLOPs |
| **Parallelism** | Expert sharding via `ParallelSplit` + `ParallelAllReduce` |

---

## 2.10 Distributed Training and Parallelism

### Problem

Train models too large for a single GPU on multi-GPU clusters.

### Solution LIFT

```
// Data parallelism
%grad_local = tensor.grad_matmul %x, %w
%grad_global = tensor.parallel_all_reduce %grad_local

// Pipeline parallelism
tensor.pipeline_send %activations, %device_1
%received = tensor.pipeline_receive %device_1

// Tensor parallelism
%shards = tensor.parallel_split %weight, %num_gpus
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | AllReduce communication volume: `2*(N-1)/N * grad_size * N_GPU` |
| **Predict** | Communication / compute ratio. If > 30%, parallelism is inefficient |
| **Energy** | 8x A100, 24h = 92.4 kWh, 37 kg CO2 |

### Parallelism Operations

`ParallelSplit`, `ParallelAllReduce`, `PipelineSend`, `PipelineReceive`, `Checkpoint`, `Offload`, `GradAccumulate`.

### Gradient Operations

`GradMatMul`, `GradReLU`, `GradSoftmax`, `GradLayerNorm`, `GradAttention`, `GradConv2D`, `GradLinear`, `GradGeLU`.

---

# Part III — Quantum Problems

## 3.1 Quantum Simulation (Hamiltonian)

### Problem

Simulate the evolution of a quantum system under a Hamiltonian: materials physics, quantum chemistry, particle physics.

### Solution LIFT

```
func @hamiltonian_sim(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Trotterisation: e^{-iHt} approximated by product of gates
  %q0a = quantum.rx %q0       // X term of the Hamiltonian
  %q1a = quantum.rx %q1
  // ZZ interaction
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c = quantum.rz %q1b      // e^{-i*J*t*ZZ}
  %q0c, %q1d = quantum.cx %q0b, %q1c
  %q2a = quantum.rx %q2
  %q3a = quantum.rx %q3
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q3c = quantum.rz %q3b
  %q2c, %q3d = quantum.cx %q2b, %q3c
  return %q0c, %q1d, %q2c, %q3d
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Verify** | Checks linearity: each qubit consumed exactly once (no-cloning) |
| **Analyse** | Counts 1Q/2Q gates, estimates fidelity and depth |
| **Optimise** | **Rotation merge**: `Rz(a)*Rz(b) → Rz(a+b)`. **Gate cancellation**: `X*X → I`, `H*H → I` |
| **Predict** | Number of shots for a given precision on the observable |
| **Topology** | Adapts the circuit to the QPU (grid, heavy-hex) with SWAP insertion |

### Available Quantum Gates

**1-qubit standard**: H, X, Y, Z, S, S†, T, T†, SX

**1-qubit parametric**: RX, RY, RZ, P, U1, U2, U3

**2-qubit**: CX (CNOT), CZ, CY, SWAP, iSWAP, ECR, RZX, XX, YY, ZZ, CP, CPhase, XY

**3-qubit**: CCX (Toffoli), CSWAP (Fredkin)

**Multi-controlled**: MCX, MCZ

**Control**: Measure, MeasureAll, Reset, Barrier, Init, Delay, VirtualRZ, IfElse

---

## 3.2 Quantum Error Correction (QEC)

### Problem

Physical qubits are noisy. To run reliable algorithms, information must be encoded in logical qubits protected by error-correcting codes.

### Solution LIFT

LIFT includes a complete QEC module (`lift-quantum::qec`):

```rust
// QEC analysis: 10 logical qubits, depth 100, surface code d=7
let analysis = QecAnalysis::analyse(
    10,                                    // logical qubits
    100,                                   // circuit depth
    QecCode::SurfaceCode { distance: 7 },  // surface code
    0.001,                                 // physical error rate 0.1%
);
// Result: 490 physical qubits, logical error ~10^-8
```

### Supported QEC Codes

| Code | Phys. qubits / logical | Distance | Error threshold | Usage |
|------|------------------------|----------|----------------|-------|
| **Surface Code** | d^2 | d | ~1% | Standard NISQ/FTQC |
| **Steane Code** | 7 | 3 | ~0.5% | Small circuits |
| **Shor Code** | 9 | 3 | ~0.3% | Educational |
| **Repetition Code** | d | d | ~3% | Bit-flip only |
| **LDPC Code** | n/k | ~sqrt(n) | ~0.8% | Reduced overhead |

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Analyse** | Computes physical qubits, logical error rate, syndrome depth |
| **Suggest** | `suggest_distance()` recommends the minimum distance for a target error rate |
| **Budget** | Verifies the QPU has enough physical qubits (e.g. IBM Eagle = 127) |

---

## 3.3 Variational Circuits (VQE)

### Problem

Find the ground state of a molecular Hamiltonian: binding energy, equilibrium geometry, electronic properties.

### Solution LIFT

```
func @vqe(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Ansatz hardware-efficient
  %q0a = quantum.ry %q0    // theta_0
  %q1a = quantum.ry %q1    // theta_1
  %q2a = quantum.ry %q2    // theta_2
  %q3a = quantum.ry %q3    // theta_3
  // Entanglement
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c, %q2b = quantum.cx %q1b, %q2a
  %q2c, %q3b = quantum.cx %q2b, %q3a
  // Parameterised layer
  %q0c = quantum.rz %q0b
  %q1d = quantum.rz %q1c
  %q2d = quantum.rz %q2c
  %q3c = quantum.rz %q3b
  return %q0c, %q1d, %q2d, %q3c
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Hybrid** | `hybrid.vqe_ansatz` manages the classical-quantum loop |
| **Gradient** | **Parameter shift**: exact gradients in 2N evaluations. **Adjoint**: 1 evaluation (simulators) |
| **Noise** | Gate-by-gate fidelity. If < threshold, recommends a shorter circuit or better QPU |
| **Reactive budget** | Stops VQE optimisation if the time budget is exhausted |

### Ansatz Types

| Type | Description | Usage |
|------|-------------|-------|
| `HardwareEfficient` | Alternating RY + CX | NISQ, short circuits |
| `StronglyEntangling` | Full entanglement | Maximum expressivity |
| `TwoLocal` | Local layers + entanglement | Trade-off |
| `UCCSD` | Unitary Coupled Cluster | Quantum chemistry |
| `Custom` | User-defined | Research |

---

## 3.4 Combinatorial Optimisation (QAOA)

### Problem

Solve NP-hard problems: maximum cut (MaxCut), travelling salesman, graph colouring, resource allocation.

### Solution LIFT

```
func @qaoa_maxcut(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Initial superposition
  %q0a = quantum.h %q0
  %q1a = quantum.h %q1
  %q2a = quantum.h %q2
  %q3a = quantum.h %q3
  // Problem layer (ZZ interactions = graph edges)
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c = quantum.rz %q1b          // gamma * edge weight (0,1)
  %q0c, %q1d = quantum.cx %q0b, %q1c
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q3c = quantum.rz %q3b          // gamma * edge weight (2,3)
  %q2c, %q3d = quantum.cx %q2b, %q3c
  // Mixer layer
  %q0d = quantum.rx %q0c          // beta
  %q1e = quantum.rx %q1d
  %q2d = quantum.rx %q2c
  %q3e = quantum.rx %q3d
  return %q0d, %q1e, %q2d, %q3e
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Hybrid** | `hybrid.qaoa_layer` stacks P layers, optimises 2P parameters (gamma, beta) |
| **Topology** | Inserts SWAPs to adapt the problem graph to the QPU topology |
| **Noise-aware** | Reorders CX gates onto qubit pairs with the best fidelity |
| **Layout mapping** | SABRE maps logical → physical qubits, minimising SWAPs |
| **Predict** | Shots needed to distinguish the best solution |

---

## 3.5 Entangled States and Quantum Protocols

### Problem

Prepare entangled states for quantum communication, teleportation, key distribution (QKD), processor benchmarking.

### Solution LIFT

```
// Bell state |Phi+> = (|00> + |11>) / sqrt(2)
func @bell_state(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
  %q0a = quantum.h %q0
  %q0b, %q1a = quantum.cx %q0a, %q1
  return %q0b, %q1a
}

// 4-qubit GHZ state
func @ghz_state(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  %q0a = quantum.h %q0
  %q0b, %q1a = quantum.cx %q0a, %q1
  %q1b, %q2a = quantum.cx %q1a, %q2
  %q2b, %q3a = quantum.cx %q2a, %q3
  return %q0b, %q1b, %q2b, %q3a
}
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Verify** | Strict linearity: each qubit consumed exactly once |
| **Analyse** | Bell = 2 gates, fidelity ~0.99. GHZ(4) = 4 gates, fidelity ~0.97 |
| **Export** | OpenQASM 3.0 executable on IBM Quantum or Amazon Braket |
| **QEC** | Computes the error-correcting code needed to protect EPR qubits |

---

# Part IV — Hybrid AI + Quantum Problems

## 4.1 Hybrid Medical Imaging (CNN + VQC)

### Problem

Classify chest X-rays (pneumonia / normal) by combining a CNN for feature extraction and a variational quantum circuit for the decision, potentially more performant on small datasets.

### Solution LIFT

```
// 1. CNN encoder (GPU): image → 4D vector
func @cnn_encoder(%img: tensor<1x1x128x128xf32>) -> tensor<1x4xf32> {
  %c1 = tensor.conv2d %img, %w1
  %r1 = tensor.relu %c1
  %p1 = tensor.maxpool2d %r1
  %c2 = tensor.conv2d %p1, %w2
  %r2 = tensor.relu %c2
  %g  = tensor.global_avgpool %r2
  %features = tensor.linear %g, %wfc
  return %features
}

// 2. Transfer + encoding
%encoded = hybrid.encode %features           // angle encoding
%qubits = hybrid.gpu_to_qpu %encoded         // GPU → QPU

// 3. VQC classifier (QPU): 4 qubits
func @vqc(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  %q0a = quantum.ry %q0
  %q1a = quantum.ry %q1
  %q2a = quantum.ry %q2
  %q3a = quantum.ry %q3
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q0c = quantum.rz %q0b
  %q1c = quantum.rz %q1b
  %q2c = quantum.rz %q2b
  %q3c = quantum.rz %q3b
  return %q0c, %q1c, %q2c, %q3c
}

// 4. Measurement + post-processing
%results = hybrid.measure_expectation %qubits
%class = hybrid.qpu_to_gpu %results
%probs = tensor.softmax %class
```

### How LIFT Exploits This (full pipeline)

| Component | Parse | Analyse | Optimise | Predict | Export |
|-----------|-------|---------|----------|---------|--------|
| **CNN** | 4D tensor types | ~10 MFLOP, ~500 KiB | Conv+relu fusion, DCE | A100: 0.001 ms | LLVM IR |
| **Interface** | GPU-QPU transfer | Encoding cost O(4) | Encode+linear fusion | Latency ~1 ms | Script |
| **VQC** | Qubit linearity | 10 gates, fidelity ~0.97 | Rotation merge | IBM: 7.9 ms | OpenQASM |

### Encoding Strategies

| Strategy | Qubits | Depth | Usage |
|----------|--------|-------|-------|
| `AngleEncoding` | N | 1 | Small vectors (< 20 features) |
| `AmplitudeEncoding` | log2(N) | N | Large vectors (compression) |
| `BasisEncoding` | N | 1 | Binary data |
| `IQPEncoding` | N | 2N | High expressivity |
| `HamiltonianEncoding` | N | N | Physics-inspired |
| `KernelEncoding` | N | 3N | Quantum kernel methods |

---

## 4.2 Drug Discovery (GNN + VQE)

### Problem

Find new therapeutic molecules by combining a GNN for rapid screening and a VQE for precise energy calculation of the best candidates.

### Solution LIFT

```
// Step 1: GNN screening (GPU) — filters 10,000 molecules
func @molecule_screen(%atoms: tensor<50x16xf32>,
                       %bonds: tensor<50x50xf32>) -> tensor<1x1xf32> {
  %msg = tensor.gnn_message_passing %atoms, %bonds, %w
  %pool = tensor.gnn_global_pooling %msg
  %score = tensor.linear %pool, %wfc
  return %score
}

// Step 2: VQE energy (QPU) — top-K molecules
%circuit = hybrid.vqe_ansatz %qubits          // UCCSD ansatz
%energy = hybrid.measure_expectation %circuit  // binding energy
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **GNN** | Filters 10,000 molecules in seconds. Optimises sparse lookups |
| **VQE** | Energy of the 10 best candidates. Reactive budget stops on convergence |
| **Gradient** | Adjoint differentiation (1 evaluation = most efficient) |
| **Noise** | Fidelity sufficient for chemical accuracy (1.6 mHartree)? |
| **QEC** | If errors too large, recommends an error-correcting code |

---

## 4.3 Quantum Finance (QAOA + Classical ML)

### Problem

Optimise an investment portfolio: classical model to predict returns + QAOA for discrete asset selection under constraints.

### Solution LIFT

```
// Step 1: Return prediction (LSTM on GPU)
func @returns(%prices: tensor<1x252x50xf32>) -> tensor<1x50xf32> {
  %h, %c = tensor.lstm_cell %prices, %h0, %c0, %w
  %returns = tensor.linear %h, %wfc
  return %returns
}

// Step 2: Portfolio selection (QAOA on QPU)
// Each qubit = decision to include an asset
// Hamiltonian = max return - min risk - budget constraint
%layer = hybrid.qaoa_layer %qubits, %gamma, %beta
%samples = hybrid.measure_samples %qubits, 4096
%best = tensor.topk %samples, 10
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **LSTM** | Predicts returns for 50 assets. Identifies memory bottleneck (cached states) |
| **QAOA** | 50 qubits = 50 assets, P=3 layers. Adapted to heavy-hex (127 qubits) |
| **Budget** | Total < 10 s (trading constraint). Split: 1 ms LSTM + 9 s QAOA |
| **Energy** | Energy cost of 4096 shots on superconducting QPU |

---

## 4.4 Quantum Machine Learning (QML)

### Problem

Use quantum kernels for classification or regression tasks where quantum feature spaces offer an advantage.

### Solution LIFT

```
// Quantum Kernel: computes similarity in Hilbert space
%encoded_x = hybrid.encode %x, "iqp"          // IQP encoding
%encoded_y = hybrid.encode %y, "iqp"
%kernel = hybrid.quantum_kernel %encoded_x, %encoded_y
%similarity = hybrid.measure_expectation %kernel

// Classification with quantum kernel
%svm_result = tensor.matmul %kernel_matrix, %alpha
%class = tensor.sigmoid %svm_result
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Encoding** | IQP encoding for maximum expressivity (depth 2N) |
| **Kernel** | `hybrid.quantum_kernel` computes Gram matrix elements |
| **Predict** | Number of shots to estimate each element with precision epsilon |
| **Analyse** | Total cost = N^2 * shots * circuit_time (N = dataset size) |

### Available Feature Maps

| Feature Map | Description | Usage |
|-------------|-------------|-------|
| `ZZFeatureMap` | ZZ interactions between features | Standard classification |
| `PauliFeatureMap` | Pauli products | High expressivity |
| `AngleEncoding` | Simple rotation | Continuous data |
| `AmplitudeEncoding` | State amplitudes | Data compression |

---

## 4.5 Materials Science (VQE + Tensor)

### Problem

Predict properties of new materials (superconductors, batteries, catalysts) by combining precise quantum simulations with ML models for screening.

### Solution LIFT

```
// Fast ML: property prediction (GPU)
func @material_screen(%composition: tensor<1x32xf32>) -> tensor<1x5xf32> {
  %h1 = tensor.linear %composition, %w1
  %a1 = tensor.gelu %h1
  %h2 = tensor.linear %a1, %w2
  %props = tensor.sigmoid %h2
  return %props                              // 5 predicted properties
}

// Precise VQE for promising candidates (QPU)
%ansatz = hybrid.vqe_ansatz %qubits          // UCCSD for chemistry
%energy = hybrid.measure_expectation %ansatz  // material energy
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Screening** | Fast MLP on GPU, filters thousands of compositions |
| **VQE** | Precise quantum computation for the best candidates |
| **Co-execution** | `hybrid.co_execute` to parallelise screening and VQE |
| **Sync** | `SyncPolicy::Pipeline` to process candidates in a stream |

---

## 4.6 Hybrid Anomaly Detection

### Problem

Detect anomalies in complex data (financial fraud, cyberattacks, industrial defects) by combining a classical autoencoder and a quantum circuit for detection in a quantum feature space.

### Solution LIFT

```
// Classical autoencoder (GPU): feature compression
func @encoder(%data: tensor<1x100xf32>) -> tensor<1x8xf32> {
  %h1 = tensor.linear %data, %w1
  %a1 = tensor.relu %h1
  %latent = tensor.linear %a1, %w2
  return %latent
}

// Quantum detection: distance measurement in Hilbert space
%encoded = hybrid.encode %latent, "angle"
%circuit = hybrid.vqc_layer %encoded
%anomaly_score = hybrid.measure_expectation %circuit
```

### How LIFT Exploits This

| Stage | LIFT Action |
|-------|-------------|
| **Encoder** | Compresses 100 features → 8 dimensions. Analyses FLOPs and memory |
| **VQC** | Measures distance in quantum space. Anomalies have a different score |
| **Co-execute** | `hybrid.co_execute` + `SyncPolicy::Asynchronous` for real-time |
| **Budget** | Latency < 10 ms for real-time detection |

---

# Part V — Cross-Cutting Capabilities

These capabilities apply to **all** problems described in Parts II, III, and IV.

## 5.1 Optimisation Pipeline (11 Passes)

LIFT provides 11 optimisation passes organised into three families:

### Classical Passes (AI)

| Pass | Description | Gain |
|------|-------------|------|
| **Canonicalize** | Simplifies `x+0 → x`, `reshape(reshape(x)) → reshape(x)`, `mul(x,1) → x` | Graph reduction |
| **Constant Folding** | Evaluates constant expressions at compile time | Fewer ops at runtime |
| **Dead Code Elimination** | Removes operations whose results are never used | -10 to -30% ops |
| **Tensor Fusion** | Fuses `conv2d + batchnorm + relu` into a single kernel | -30% memory, -20% latency |
| **Flash Attention** | Replaces standard O(n^2) attention with FlashAttention O(n) | -90% memory for long sequences |
| **Common Subexpr Elim** | Detects and eliminates redundant computations | Variable |
| **Quantisation Pass** | Annotates INT8/INT4/FP8-compatible ops | 2-8x less memory |

### Quantum Passes

| Pass | Description | Gain |
|------|-------------|------|
| **Gate Cancellation** | Removes self-inverse pairs: `H*H → I`, `X*X → I` | Fewer gates |
| **Rotation Merge** | Fuses consecutive rotations: `Rz(a)*Rz(b) → Rz(a+b)` | Reduced depth |
| **Noise-Aware Schedule** | Reorders 2Q gates onto high-fidelity qubit pairs | Better fidelity |
| **Layout Mapping** | SABRE algorithm: maps logical → physical qubits, minimises SWAPs | Executable circuit |

### CLI Usage

```bash
lift optimise model.lif --config config.lith -o optimised.lif
```

### Programmatic Usage (Rust)

```rust
let mut pm = PassManager::new();
pm.add_pass(Box::new(Canonicalize));
pm.add_pass(Box::new(TensorFusion));
pm.add_pass(Box::new(DeadCodeElimination));
pm.add_pass(Box::new(GateCancellation));
pm.add_pass(Box::new(RotationMerge));
let results = pm.run_all(&mut ctx);
// results: Vec<(String, PassResult)>  — Changed / Unchanged / Error
```

---

## 5.2 Performance Prediction

### Roofline Model (Classical GPU)

LIFT models two NVIDIA GPUs with precise parameters:

| GPU | TFLOPS FP16 | Bandwidth | VRAM |
|-----|-------------|-----------|------|
| **A100** | 312 TFLOPS | 2039 GB/s | 80 GB |
| **H100** | 989 TFLOPS | 3350 GB/s | 80 GB |

For each model, LIFT computes:

- **Compute time** = FLOPs / TFLOPS
- **Memory time** = Bytes / Bandwidth
- **Predicted time** = max(compute, memory) → identifies the bottleneck
- **Arithmetic intensity** = FLOPs / Bytes → compared to the crossover point
- **Number of GPUs** = ceil(total_memory / VRAM_per_GPU)

### Quantum Model

LIFT models three types of QPU:

| QPU | 1Q Time | 2Q Time | 1Q Fidelity | 2Q Fidelity | Qubits |
|-----|---------|---------|-------------|-------------|--------|
| **Superconducting** (IBM) | 0.02 us | 0.3 us | 99.9% | 99.0% | 127 |
| **Trapped ions** (IonQ) | 10 us | 200 us | 99.97% | 99.5% | 32 |
| **Neutral atoms** | 1 us | 5 us | 99.5% | 98.0% | 256 |

For each circuit, LIFT computes:

- **Estimated fidelity** = prod(fidelity_1Q^n1Q * fidelity_2Q^n2Q)
- **Circuit time** = n1Q * t1Q + n2Q * t2Q + nMeas * tMeas
- **Required shots** = 1 / (precision^2 * fidelity^2)
- **Total time** = shots * circuit_time

### CLI Usage

```bash
lift predict model.lif --device a100 --quantum-device ibm_kyoto
```

---

## 5.3 Quantum Noise Modelling

LIFT models 8 types of quantum noise:

| Noise Model | Fidelity Formula | Usage |
|-------------|-----------------|-------|
| **Ideal** | F = 1.0 | Reference |
| **Depolarising** | F = 1 - p | Generic noise |
| **Amplitude damping** | F = 1 - gamma/2 | T1 decay |
| **Phase damping** | F = 1 - gamma/2 | T2 dephasing |
| **Bit-flip** | F = 1 - p | Classical error |
| **Phase-flip** | F = 1 - p | Phase error |
| **Thermal relaxation** | F = (1 + e^{-t/T1} + 2*e^{-t/T2}) / 4 | Realistic (IBM) |
| **Kraus** | F ≈ 0.99 (approx.) | General quantum channel |

### Kraus Channels

LIFT provides a complete Kraus channel algebra (`lift-quantum::kraus`):

- **Depolarising**: `KrausChannel::depolarizing(p, n_qubits)`
- **Amplitude damping**: `KrausChannel::amplitude_damping(gamma)`
- **Phase damping**: `KrausChannel::phase_damping(lambda)`
- **Pauli channel**: `KrausChannel::pauli(px, py, pz)`
- **Composition**: `channel1.compose(&channel2)`
- **Average fidelity**: `channel.average_gate_fidelity()`

### Gate-by-Gate Noise Tracking

```rust
let mut circuit = CircuitNoise::new();
let g1q = GateNoise::with_depolarizing(0.999, 0.02);  // 1Q gate
let g2q = GateNoise::with_depolarizing(0.99, 0.3);    // 2Q gate

circuit.add_gate(&g1q, false);  // RY
circuit.add_gate(&g2q, true);   // CX — dominant error source
// circuit.total_fidelity, circuit.gate_count, circuit.meets_threshold(0.90)
```

---

## 5.4 Quantum Processor Topology

LIFT models 5 QPU topologies:

| Topology | Manufacturer | Qubits | Connectivity |
|----------|-------------|--------|-------------|
| **Grid** (n x m) | Google Sycamore | n*m | 4 neighbours max |
| **Heavy-hex** | IBM Eagle/Osprey | 127 | 2-3 neighbours |
| **All-to-all** | IonQ | variable | Fully connected |
| **Linear** | Simple chain | variable | 2 neighbours |
| **Tree** | Hierarchical | variable | log(n) depth |

### Features

For each topology, LIFT provides:

- **Connectivity**: `are_connected(q0, q1)` — are two qubits neighbours?
- **Shortest path**: `shortest_path(from, to)` — BFS on the graph
- **SWAP distance**: `swap_distance(from, to)` — number of SWAPs required
- **Neighbours**: `neighbors(q)` — adjacent qubits
- **Diameter**: `diameter()` — longest shortest path
- **Average connectivity**: `avg_connectivity()` — mean degree

### Impact on Compilation

The topology determines the cost of **layout mapping** (SABRE pass):

| Topology | SWAP q0→q3 (4 qubits) | Fidelity Impact |
|----------|----------------------|-----------------|
| All-to-all | 0 SWAP | None |
| Grid 2x2 | 1 SWAP | 3 additional CX gates |
| Linear | 2 SWAP | 6 additional CX gates |
| Heavy-hex | Variable | Depends on placement |

---

## 5.5 Energy Estimation and Carbon Footprint

LIFT estimates energy consumption at three levels:

### GPU Inference

```
Energy (J) = TDP (W) * PUE * time (s) * n_GPU
CO2 (g) = Energy (kWh) * emission_factor (g/kWh)
```

| GPU | TDP | PUE | Emission Factor |
|-----|-----|-----|----------------|
| **A100** | 400 W | 1.1 | 400 g CO2/kWh |
| **H100** | 700 W | 1.1 | 400 g CO2/kWh |

### Training (cluster)

Example computed by LIFT: **8x A100 for 24 hours**:
- Energy = 8 * 400 W * 1.1 * 86400 s = 92.4 kWh
- CO2 = 92.4 * 400 = 36,960 g = **37 kg CO2**

### Quantum Execution

The energy cost of a superconducting QPU is dominated by **cryogenics**:
- Cryostat power: ~25 kW (to maintain 15 mK)
- Power per qubit: negligible vs. cryostat
- `energy.quantum_energy_joules(circuit_time_us, n_qubits)`: includes cryogenic cost

---

## 5.6 Budget and Deployment Constraints

### Static Budget

Checks hard constraints before compilation:

```rust
let budget = Budget {
    max_flops: Some(10_000_000_000),       // 10 GFLOP max
    max_memory_bytes: Some(80_000_000_000), // 80 GB VRAM
    max_time_ms: Some(100.0),              // 100 ms max latency
    min_fidelity: Some(0.90),              // 90% min fidelity
    max_circuit_depth: None,
};

budget.check_flops(report.total_flops)?;      // Ok or Err
budget.check_memory(report.total_memory)?;     // Ok or Err
budget.check_fidelity(analysis.fidelity)?;     // Ok or Err
```

### Reactive Budget

Tracks consumption in real time during iterative loops (VQE, QAOA):

```rust
let mut tracker = ReactiveBudget::new(budget);

for i in 0..max_iterations {
    tracker.consume(flops, memory, time_ms, fidelity_decay);
    if tracker.check_remaining().is_err() {
        println!("Budget exhausted at iteration {}", i);
        break;
    }
}

let util = tracker.utilisation();
// util.time_ratio, util.flops_ratio, util.memory_ratio
```

---

## 5.7 Export to Backends

### LLVM IR (GPU / CPU)

```bash
lift export model.lif --backend llvm -o model.ll
```

LIFT generates LLVM IR that can be compiled by `llc` to:
- **CUDA PTX**: execution on NVIDIA GPUs
- **x86-64**: execution on CPU (with AVX-512 if available)
- **ARM**: embedded deployment

### OpenQASM 3.0 (QPU)

```bash
lift export model.lif --backend qasm -o circuit.qasm
```

LIFT generates OpenQASM 3.0 compatible with:
- **IBM Quantum** (Qiskit Runtime)
- **Amazon Braket**
- **Azure Quantum**
- **Simulators** (Qiskit Aer, Cirq)

### Programmatic Usage

```rust
// Export LLVM
let llvm_exporter = LlvmExporter::new();
let llvm_ir = llvm_exporter.export(&ctx)?;

// Export OpenQASM
let qasm_exporter = QasmExporter::new();
let qasm_code = qasm_exporter.export(&ctx)?;
```

---

## 5.8 Configuration (.lith)

The `.lith` file configures the LIFT pipeline using a simple INI format:

```ini
[target]
backend = llvm
device = A100

[budget]
max_flops = 10000000000
max_memory_bytes = 80000000000
min_fidelity = 0.90

[optimisation]
level = O3
passes = canonicalize,tensor_fusion,dce,gate_cancellation,rotation_merge

[simulation]
enabled = true
noise_model = depolarizing

[quantum]
topology = grid
num_qubits = 4
shots = 4096
error_rate = 0.001
```

### Optimisation Levels

| Level | Enabled Passes |
|-------|---------------|
| **O0** | None (raw IR) |
| **O1** | Canonicalize, DCE |
| **O2** | + Tensor fusion, Gate cancellation, Rotation merge |
| **O3** | + Flash attention, CSE, Noise-aware schedule, Layout mapping, Quantisation |

---

# Part VI — Quick Reference

## 6.1 Catalogue of 90+ Tensor Operations

### Arithmetic

| Operation | `.lif` Syntax | Description |
|-----------|--------------|-------------|
| `Add` | `tensor.add` | Element-wise addition |
| `Sub` | `tensor.sub` | Subtraction |
| `Mul` | `tensor.mul` | Element-wise multiplication |
| `Div` | `tensor.div` | Division |
| `Neg` | `tensor.neg` | Negation |
| `MatMul` | `tensor.matmul` | Matrix product |
| `Linear` | `tensor.linear` | Linear layer (matmul + bias) |
| `Conv2D` | `tensor.conv2d` | 2D convolution |
| `Embedding` | `tensor.embedding` | Embedding table lookup |

### Activations

| Operation | `.lif` Syntax | Formula |
|-----------|--------------|---------|
| `ReLU` | `tensor.relu` | max(0, x) |
| `GeLU` | `tensor.gelu` | x * Phi(x) |
| `SiLU` | `tensor.silu` | x * sigmoid(x) |
| `Sigmoid` | `tensor.sigmoid` | 1 / (1 + e^-x) |
| `Softmax` | `tensor.softmax` | e^x_i / sum(e^x_j) |
| `Tanh` | `tensor.tanh` | (e^x - e^-x) / (e^x + e^-x) |
| `LeakyReLU` | `tensor.leaky_relu` | max(alpha*x, x) |
| `ELU` | `tensor.elu` | x if x>0, alpha*(e^x-1) otherwise |
| `Mish` | `tensor.mish` | x * tanh(softplus(x)) |
| `HardSwish` | `tensor.hard_swish` | x * relu6(x+3) / 6 |
| `HardSigmoid` | `tensor.hard_sigmoid` | relu6(x+3) / 6 |

### Normalisation

| Operation | `.lif` Syntax | Usage |
|-----------|--------------|-------|
| `LayerNorm` | `tensor.layernorm` | Transformers |
| `RMSNorm` | `tensor.rmsnorm` | LLaMA, Mistral |
| `BatchNorm` | `tensor.batchnorm` | CNN (training) |
| `GroupNorm` | `tensor.groupnorm` | Diffusion models |
| `InstanceNorm` | `tensor.instancenorm` | Style transfer |

### Shape

| Operation | `.lif` Syntax | Zero-FLOP |
|-----------|--------------|-----------|
| `Reshape` | `tensor.reshape` | Yes |
| `Transpose` | `tensor.transpose` | Yes |
| `Concat` | `tensor.concat` | Yes |
| `Split` | `tensor.split` | Yes |
| `Gather` | `tensor.gather` | Yes |
| `Scatter` | `tensor.scatter` | Yes |
| `Squeeze` | `tensor.squeeze` | Yes |
| `Unsqueeze` | `tensor.unsqueeze` | Yes |
| `Permute` | `tensor.permute` | Yes |
| `Expand` | `tensor.expand` | Yes |
| `Slice` | `tensor.slice` | Yes |
| `Pad` | `tensor.pad` | Yes |
| `Tile` | `tensor.tile` | Yes |

### Attention (8 variants)

| Operation | `.lif` Syntax |
|-----------|---------------|
| `Attention` | `tensor.attention` |
| `MultiHeadAttention` | `tensor.multi_head_attention` |
| `MultiQueryAttention` | `tensor.multi_query_attention` |
| `GroupedQueryAttention` | `tensor.grouped_query_attention` |
| `FlashAttention` | `tensor.flash_attention` |
| `SlidingWindowAttention` | `tensor.sliding_window_attention` |
| `CrossAttention` | `tensor.cross_attention` |
| `PagedAttention` | `tensor.paged_attention` |

### Convolution (6 variants)

| Operation | `.lif` Syntax |
|-----------|---------------|
| `Conv1D` | `tensor.conv1d` |
| `Conv2D` | `tensor.conv2d` |
| `Conv3D` | `tensor.conv3d` |
| `ConvTranspose2D` | `tensor.conv_transpose2d` |
| `DepthwiseConv2D` | `tensor.depthwise_conv2d` |
| `DilatedConv2D` | `tensor.dilated_conv2d` |

### Pooling

| Operation | `.lif` Syntax |
|-----------|---------------|
| `MaxPool2D` | `tensor.maxpool2d` |
| `AvgPool2D` | `tensor.avgpool2d` |
| `AdaptiveAvgPool2D` | `tensor.adaptive_avgpool2d` |
| `GlobalAvgPool` | `tensor.global_avgpool` |

### Recurrent

| Operation | `.lif` Syntax |
|-----------|---------------|
| `LSTMCell` | `tensor.lstm_cell` |
| `GRUCell` | `tensor.gru_cell` |
| `RNNCell` | `tensor.rnn_cell` |

### Advanced Mathematics

| Operation | `.lif` Syntax | Complexity |
|-----------|---------------|-----------|
| `Einsum` | `tensor.einsum` | Variable |
| `FFT` | `tensor.fft` | O(n log n) |
| `IFFT` | `tensor.ifft` | O(n log n) |
| `SVD` | `tensor.svd` | O(mn min(m,n)) |
| `Eig` | `tensor.eig` | O(n^3) |
| `Solve` | `tensor.solve` | O(n^3) |
| `TopK` | `tensor.topk` | O(n log k) |
| `Sort` | `tensor.sort` | O(n log n) |
| `Cumsum` | `tensor.cumsum` | O(n) |
| `Where` | `tensor.where` | O(n) |
| `Clamp` | `tensor.clamp` | O(n) |

### Sparse, Quantisation, Diffusion, GNN, MoE

| Operation | `.lif` Syntax |
|-----------|---------------|
| `SparseMatMul` | `tensor.sparse_matmul` |
| `SparseEmbedding` | `tensor.sparse_embedding` |
| `Quantize` | `tensor.quantize` |
| `Dequantize` | `tensor.dequantize` |
| `QuantizeInt4` | `tensor.quantize_int4` |
| `DequantizeInt4` | `tensor.dequantize_int4` |
| `QuantizeFp8` | `tensor.quantize_fp8` |
| `DequantizeFp8` | `tensor.dequantize_fp8` |
| `UNetDownBlock` | `tensor.unet_down_block` |
| `UNetUpBlock` | `tensor.unet_up_block` |
| `TimestepEmbedding` | `tensor.timestep_embedding` |
| `GNNMessagePassing` | `tensor.gnn_message_passing` |
| `GNNGlobalPooling` | `tensor.gnn_global_pooling` |
| `MoEDispatch` | `tensor.moe_dispatch` |
| `MoECombine` | `tensor.moe_combine` |

### Memory, Gradient, Parallelism, Fused

| Operation | `.lif` Syntax |
|-----------|---------------|
| `Checkpoint` | `tensor.checkpoint` |
| `Offload` | `tensor.offload` |
| `GradAccumulate` | `tensor.grad_accumulate` |
| `GradMatMul` | `tensor.grad_matmul` |
| `GradReLU` | `tensor.grad_relu` |
| `GradSoftmax` | `tensor.grad_softmax` |
| `GradLayerNorm` | `tensor.grad_layernorm` |
| `GradAttention` | `tensor.grad_attention` |
| `GradConv2D` | `tensor.grad_conv2d` |
| `GradLinear` | `tensor.grad_linear` |
| `GradGeLU` | `tensor.grad_gelu` |
| `ParallelSplit` | `tensor.parallel_split` |
| `ParallelAllReduce` | `tensor.parallel_all_reduce` |
| `PipelineSend` | `tensor.pipeline_send` |
| `PipelineReceive` | `tensor.pipeline_receive` |
| `FusedMatMulBiasReLU` | `tensor.fused_matmul_bias_relu` |
| `FusedMatMulBias` | `tensor.fused_matmul_bias` |
| `FusedLinearGeLU` | `tensor.fused_linear_gelu` |
| `FusedAttentionLayerNorm` | `tensor.fused_attention_layernorm` |
| `FusedLinearSiLU` | `tensor.fused_linear_silu` |
| `FusedConvBatchNormReLU` | `tensor.fused_conv_batchnorm_relu` |

### Constants

| Operation | `.lif` Syntax |
|-----------|---------------|
| `Constant` | `tensor.constant` |
| `Zeros` | `tensor.zeros` |
| `Ones` | `tensor.ones` |
| `Arange` | `tensor.arange` |
| `Full` | `tensor.full` |

---

## 6.2 Catalogue of 50+ Quantum Gates

### 1-Qubit Standard

| Gate | `.lif` Syntax | Clifford | Self-Inverse | Parametric |
|------|--------------|----------|-------------|------------|
| H | `quantum.h` | Yes | Yes | No |
| X | `quantum.x` | Yes | Yes | No |
| Y | `quantum.y` | Yes | Yes | No |
| Z | `quantum.z` | Yes | Yes | No |
| S | `quantum.s` | Yes | No | No |
| S† | `quantum.sdg` | Yes | No | No |
| T | `quantum.t` | No | No | No |
| T† | `quantum.tdg` | No | No | No |
| SX | `quantum.sx` | Yes | No | No |

### 1-Qubit Parametric

| Gate | `.lif` Syntax | Parameters |
|------|--------------|------------|
| RX | `quantum.rx` | theta |
| RY | `quantum.ry` | theta |
| RZ | `quantum.rz` | theta |
| P | `quantum.p` | phi |
| U1 | `quantum.u1` | lambda |
| U2 | `quantum.u2` | phi, lambda |
| U3 | `quantum.u3` | theta, phi, lambda |

### 2-Qubit

| Gate | `.lif` Syntax | Entangling | Native On |
|------|--------------|-----------|----------|
| CX (CNOT) | `quantum.cx` | Yes | IBM |
| CZ | `quantum.cz` | Yes | Google |
| CY | `quantum.cy` | Yes | — |
| SWAP | `quantum.swap` | Yes | — |
| iSWAP | `quantum.iswap` | Yes | Google |
| ECR | `quantum.ecr` | Yes | IBM Eagle |
| RZX | `quantum.rzx` | Yes | — |
| XX | `quantum.xx` | Yes | IonQ |
| YY | `quantum.yy` | Yes | — |
| ZZ | `quantum.zz` | Yes | — |
| CP | `quantum.cp` | Yes | — |
| CPhase | `quantum.cphase` | Yes | Rigetti |
| XY | `quantum.xy` | Yes | Rigetti |

### 3-Qubit and Multi-Controlled

| Gate | `.lif` Syntax | Qubits |
|------|--------------|--------|
| CCX (Toffoli) | `quantum.ccx` | 3 |
| CSWAP (Fredkin) | `quantum.cswap` | 3 |
| MCX | `quantum.mcx` | N |
| MCZ | `quantum.mcz` | N |

### Control and Measurement

| Gate | `.lif` Syntax | Description |
|------|--------------|-------------|
| Measure | `quantum.measure` | Measures a qubit |
| MeasureAll | `quantum.measure_all` | Measures all qubits |
| Reset | `quantum.reset` | Resets a qubit to \|0> |
| Barrier | `quantum.barrier` | Prevents reordering |
| Init | `quantum.init` | Initialises a register |
| Delay | `quantum.delay` | Time delay |
| VirtualRZ | `quantum.virtual_rz` | Virtual rotation (zero cost) |
| IfElse | `quantum.if_else` | Classical conditional branching |

### Native Gate Sets by Manufacturer

| Manufacturer | Native Gates |
|-------------|---------------|
| **IBM Eagle** | RZ, SX, X, CX, ECR |
| **IBM Kyoto** | RZ, SX, X, CX, ECR |
| **Rigetti** | RX, RZ, CPhase, XY |
| **IonQ** | GPI, GPI2, MS |
| **Quantinuum** | RZ, RX, ZZ |

---

## 6.3 Catalogue of Hybrid Operations

### Encoding / Decoding

| Operation | `.lif` Syntax | Description |
|-----------|--------------|-------------|
| `Encode` | `hybrid.encode` | Encodes classical data into a quantum state |
| `Decode` | `hybrid.decode` | Extracts classical data from a quantum state |

### Gradient Methods

| Operation | `.lif` Syntax | Evaluations | Exact |
|-----------|--------------|-------------|-------|
| `ParameterShift` | `hybrid.parameter_shift` | 2N | Yes |
| `FiniteDifference` | `hybrid.finite_difference` | N+1 | No |
| `SPSA` | `hybrid.spsa` | 2 | No |
| `AdjointDiff` | `hybrid.adjoint_diff` | 1 | Yes |
| `StochasticParamShift` | `hybrid.stochastic_param_shift` | 2 | No |
| `JointGradient` | `hybrid.joint_gradient` | Variable | Mixed |

### Variational Algorithms

| Operation | `.lif` Syntax | Usage |
|-----------|--------------|-------|
| `VqcLayer` | `hybrid.vqc_layer` | Generic variational circuit |
| `VqeAnsatz` | `hybrid.vqe_ansatz` | Variational eigensolver (chemistry) |
| `QaoaLayer` | `hybrid.qaoa_layer` | Combinatorial optimisation |
| `QuantumKernel` | `hybrid.quantum_kernel` | Quantum kernel (SVM) |

### Data Transfer

| Operation | `.lif` Syntax | Direction |
|-----------|--------------|-----------|
| `GpuToQpu` | `hybrid.gpu_to_qpu` | GPU → QPU |
| `QpuToGpu` | `hybrid.qpu_to_gpu` | QPU → GPU |

### Processing and Measurement

| Operation | `.lif` Syntax | Description |
|-----------|--------------|-------------|
| `ClassicalPreprocess` | `hybrid.classical_preprocess` | Classical preprocessing |
| `QuantumPostprocess` | `hybrid.quantum_postprocess` | Quantum postprocessing |
| `HybridForward` | `hybrid.forward` | Hybrid forward pass |
| `HybridBackward` | `hybrid.backward` | Hybrid backward pass |
| `CoExecute` | `hybrid.co_execute` | GPU + QPU co-execution |
| `MeasureExpectation` | `hybrid.measure_expectation` | Observable expectation value |
| `MeasureSamples` | `hybrid.measure_samples` | Measurement samples |

---

# Summary — The Value of LIFT

| Dimension | Without LIFT | With LIFT |
|-----------|-------------|----------|
| **Languages** | Python (AI) + Qiskit (quantum) + CUDA (GPU) | A single `.lif` file |
| **Optimisation** | Manual, framework-specific | 11 automatic passes |
| **Verification** | Manual tests, runtime errors | SSA + types + linearity at compile time |
| **Performance** | Empirical benchmarks | Roofline prediction + quantum model |
| **Noise** | Ignored or modelled separately | Integrated into the compilation pipeline |
| **Energy** | Unknown | Automatic estimation (GPU + QPU + CO2) |
| **Budget** | No constraints | Static + reactive, automatic stopping |
| **Export** | Manual conversion between formats | LLVM IR + OpenQASM 3.0 in one command |
| **Topology** | Manual adaptation to QPU | Automatic layout mapping (SABRE) |

**LIFT transforms a hybrid description into a reliable, optimised, and predictable production system, in a single tool.**
