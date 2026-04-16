# LIFT Integration Test: Hybrid AI+Quantum Medical Image Classification

## Real-World Problem

Hospitals and diagnostic centres must rapidly analyse thousands of chest X-rays to
detect pathologies (pneumonia, COVID-19, tumours). Classical CNN models achieve good
accuracy but demand large annotated datasets and heavy compute. Variational quantum
circuits (VQC) could improve generalisation on small datasets, but current frameworks
(PennyLane, Qiskit) cannot jointly optimise the classical and quantum parts within a
single compiler, nor accurately simulate real QPU noise before deployment.

## Solution with LIFT

LIFT unifies the hybrid model in one IR, optimises both parts simultaneously, predicts
performance and compiles to GPU (for the CNN) and QPU (for the quantum circuit).

### Model Architecture

1. **Classical encoder** — CNN (`conv2d`, `relu`, `maxpool2d`, `global_avgpool`, `linear`) → 4-feature vector
2. **Quantum encoding** — angle encoding of 4 features → 4 qubits
3. **Parametrised circuit** (VQC) — `RY` + `CX` layers → `RZ` rotations → expectation measurement
4. **Post-processing** — `linear` + `softmax` → 2-class output (pneumonia / normal)

### What This Test Validates

| # | Capability Tested | LIFT Crates Used |
|---|-------------------|------------------|
| 1 | IR construction (types, ops, blocks) | `lift-core` |
| 2 | `.lif` parsing and IR building | `lift-ast` |
| 3 | Tensor shape inference and FLOPs | `lift-tensor` |
| 4 | Quantum gate properties and noise | `lift-quantum` |
| 5 | Hybrid encoding and gradient methods | `lift-hybrid` |
| 6 | Verification (SSA, linearity) | `lift-core::verifier` |
| 7 | Optimisation passes (fusion, flash, gate cancel) | `lift-opt` |
| 8 | Resource analysis (FLOPs, memory, gates) | `lift-sim` |
| 9 | Roofline performance prediction | `lift-predict` |
| 10 | Noise modelling and fidelity estimation | `lift-quantum::noise` |
| 11 | Device topology and routing | `lift-quantum::topology` |
| 12 | Energy and carbon estimation | `lift-sim::cost` |
| 13 | Budget enforcement (static + reactive) | `lift-sim::cost` |
| 14 | Configuration parsing (`.lith`) | `lift-config` |
| 15 | Export to LLVM IR and OpenQASM 3.0 | `lift-export` |

### Benefits for a Hospital / Diagnostic Centre

- **Inference latency < 10 ms** per image via hybrid fusion + FlashAttention
- **Cost savings**: accurate memory prediction avoids GPU over-provisioning
- **Reliability**: noise simulation before QPU avoids wasted runs (~$10-50 each on IBM Kyoto)
- **Carbon traceability**: integrated CO2 reports for ESG compliance

## Build and Run

```bash
cd lift-test
cargo run
```
