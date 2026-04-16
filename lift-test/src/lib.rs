// ============================================================================
// LIFT Integration Test — Library Root
// ============================================================================
//
// This crate rigorously tests the LIFT compiler framework on a real-world
// problem: hybrid AI+Quantum classification of chest X-rays (pneumonia
// detection). The architecture mirrors LIFT's own 6-step compilation pipeline:
//
//   Step 1 — Parse & Verify    (step1_parse)
//   Step 2 — Static Analysis   (step2_analyse)
//   Step 3 — Optimisation      (step3_optimise)
//   Step 4 — Prediction        (step4_predict)
//   Step 5 — Export            (step5_export)
//   Step 6 — Feedback / Budget (step6_feedback)
//
// Supporting modules:
//   ir_builder — Programmatic IR construction for CNN encoder and VQC circuit
//   config     — .lith configuration parsing and validation
//   report     — Test harness, result tracking, formatting utilities
//
// ============================================================================

/// Step 1: Parse `.lif` files and verify IR correctness (SSA, types, linearity).
pub mod step1_parse;

/// Step 2: Static analysis — FLOPs, memory, gate counts, fidelity estimation.
pub mod step2_analyse;

/// Step 3: Optimisation passes — classical (fusion, DCE), quantum (gate cancel,
/// rotation merge), and hybrid (layout mapping, noise-aware scheduling).
pub mod step3_optimise;

/// Step 4: Performance prediction — GPU roofline model and quantum fidelity/shot
/// estimation across multiple hardware targets.
pub mod step4_predict;

/// Step 5: Export to backend code — LLVM IR (GPU/CPU) and OpenQASM 3.0 (QPU).
pub mod step5_export;

/// Step 6: Feedback loop — budget enforcement (static + reactive), energy/CO₂
/// estimation, and constraint validation for production deployment.
pub mod step6_feedback;

/// Programmatic IR construction for the CNN encoder and VQC circuit.
pub mod ir_builder;

/// `.lith` configuration parsing and validation.
pub mod config;

/// Test harness: result tracking, step formatting, FLOP/byte display.
pub mod report;
