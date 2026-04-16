// ============================================================================
// step2_analyse.rs — Step 2: Static Analysis (Simulation Without Execution)
// ============================================================================
//
// LIFT traverses the IR graph and computes resource metrics without running
// any code:
//
//   - **FLOPs**: exact formulas for each tensor op (conv2d, matmul, etc.).
//   - **Memory**: size of intermediate tensors, peak VRAM via SSA liveness.
//   - **Quantum depth**: number of gates on the critical path.
//   - **Estimated fidelity**: noise model of the target QPU composed across
//     every gate.
//   - **Energy**: estimated power draw (GPU + cryogenics).
//
// If the user defined a budget in `.lith`, LIFT checks constraints now.
//
// Equivalent CLI:
//   lift analyse pneumonia.lif
//
// ============================================================================

use lift_core::Context;
use lift_core::types::{DataType, Dimension, MemoryLayout, TensorTypeInfo};
use lift_sim::AnalysisReport;
use lift_sim::QuantumAnalysis;

use crate::report::{self, TestReport};

// ────────────────────────────────────────────────────────────────────────────
// Classical analysis (tensor dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Analyse the CNN context: FLOPs, memory, peak memory, op breakdown.
pub fn analyse_cnn(ctx: &Context, report: &mut TestReport) -> AnalysisReport {
    let r = lift_sim::analyze_module(ctx);

    println!("    CNN Analysis:");
    println!("      Total ops:    {}", r.num_ops);
    println!("      Tensor ops:   {}", r.num_tensor_ops);
    println!("      Total FLOPs:  {}", report::format_flops(r.total_flops));
    println!("      Total memory: {}", report::format_bytes(r.total_memory_bytes));
    println!("      Peak memory:  {}", report::format_bytes(r.peak_memory_bytes));

    let mut ops: Vec<_> = r.op_breakdown.iter().collect();
    ops.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in &ops {
        println!("        {}: {}", name, count);
    }

    report.check("CNN FLOPs > 0", r.total_flops > 0);
    report.check("CNN memory > 0", r.total_memory_bytes > 0);
    report.check("CNN has tensor ops", r.num_tensor_ops > 0);

    r
}

// ────────────────────────────────────────────────────────────────────────────
// Quantum analysis (quantum dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Analyse the VQC context: gate count, qubit usage, fidelity estimate.
pub fn analyse_vqc(ctx: &Context, report: &mut TestReport) -> QuantumAnalysis {
    let q = lift_sim::analyze_quantum_ops(ctx);

    println!("    VQC Quantum Analysis:");
    println!("      Qubits used:    {}", q.num_qubits_used);
    println!("      Gate count:     {}", q.gate_count);
    println!("      1Q gates:       {}", q.one_qubit_gates);
    println!("      2Q gates:       {}", q.two_qubit_gates);
    println!("      Est. fidelity:  {:.6}", q.estimated_fidelity);

    report.check("VQC gate count > 0", q.gate_count > 0);
    report.check("VQC has 1Q gates", q.one_qubit_gates > 0);
    report.check("VQC has 2Q gates", q.two_qubit_gates > 0);
    report.check(
        "VQC fidelity in (0, 1]",
        q.estimated_fidelity > 0.0 && q.estimated_fidelity <= 1.0,
    );

    q
}

// ────────────────────────────────────────────────────────────────────────────
// Shape inference and FLOPs (lift-tensor)
// ────────────────────────────────────────────────────────────────────────────

/// Test shape inference, FLOPs counting, and operation categorisation
/// using `lift_tensor::ShapeInference` and `lift_tensor::TensorOp`.
pub fn test_shape_inference(report: &mut TestReport) {
    use lift_tensor::{ShapeInference, TensorOp};

    // ── MatMul: [1×64] @ [64×4] ──
    let a = TensorTypeInfo {
        shape: vec![Dimension::Constant(1), Dimension::Constant(64)],
        dtype: DataType::FP32,
        layout: MemoryLayout::Contiguous,
    };
    let b = TensorTypeInfo {
        shape: vec![Dimension::Constant(64), Dimension::Constant(4)],
        dtype: DataType::FP32,
        layout: MemoryLayout::Contiguous,
    };

    let shape_result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]);
    if let Ok(shapes) = &shape_result {
        println!(
            "    MatMul [1×64] @ [64×4] → {:?}",
            shapes[0]
                .shape
                .iter()
                .map(|d| format!("{:?}", d))
                .collect::<Vec<_>>()
        );
    }
    report.check("MatMul shape inference", shape_result.is_ok());

    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
    println!("    MatMul FLOPs: {:?}", flops);
    report.check("MatMul FLOPs computation", flops.is_some());

    let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
    println!("    MatMul memory: {:?} bytes", mem);
    report.check("MatMul memory estimation", mem.is_some());

    // ── Conv2d: [1,1,128,128] * [32,1,3,3] ──
    let img = TensorTypeInfo {
        shape: vec![
            Dimension::Constant(1),
            Dimension::Constant(1),
            Dimension::Constant(128),
            Dimension::Constant(128),
        ],
        dtype: DataType::FP32,
        layout: MemoryLayout::Contiguous,
    };
    let kernel = TensorTypeInfo {
        shape: vec![
            Dimension::Constant(32),
            Dimension::Constant(1),
            Dimension::Constant(3),
            Dimension::Constant(3),
        ],
        dtype: DataType::FP32,
        layout: MemoryLayout::Contiguous,
    };
    let conv_flops = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&img, &kernel]);
    println!("    Conv2d FLOPs: {:?}", conv_flops);
    report.check("Conv2d FLOPs computation", conv_flops.is_some());

    // ── Op categorisation ──
    report.check("Reshape is zero-FLOP", TensorOp::Reshape.is_zero_flop());
    report.check("MatMul is NOT zero-FLOP", !TensorOp::MatMul.is_zero_flop());
    report.check("ReLU is activation", TensorOp::ReLU.is_activation());
    report.check("Attention is attention op", TensorOp::Attention.is_attention());
    report.check("Conv2D is convolution", TensorOp::Conv2D.is_convolution());
    report.check(
        "FusedMatMulBiasReLU is fused",
        TensorOp::FusedMatMulBiasReLU.is_fused(),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Quantum gate properties (lift-quantum)
// ────────────────────────────────────────────────────────────────────────────

/// Test quantum gate metadata: arity, Clifford membership, parametric flags,
/// name round-trips, and hardware-native gate sets.
pub fn test_quantum_gates(report: &mut TestReport) {
    use lift_quantum::gates::Provider;
    use lift_quantum::QuantumGate;

    // ── Single-gate properties ──
    let h = QuantumGate::H;
    println!(
        "    H: name={}, qubits={}, clifford={}, self_inv={}",
        h.op_name(),
        h.num_qubits(),
        h.is_clifford(),
        h.is_self_inverse()
    );
    report.check("H is 1-qubit", h.num_qubits() == 1);
    report.check("H is Clifford", h.is_clifford());
    report.check("H is self-inverse", h.is_self_inverse());

    let cx = QuantumGate::CX;
    println!(
        "    CX: name={}, qubits={}, entangling={}",
        cx.op_name(),
        cx.num_qubits(),
        cx.is_entangling()
    );
    report.check("CX is 2-qubit", cx.num_qubits() == 2);
    report.check("CX is entangling", cx.is_entangling());

    report.check("RY is parametric", QuantumGate::RY.is_parametric());
    report.check("RY is 1-qubit", QuantumGate::RY.num_qubits() == 1);
    report.check("RZ is parametric", QuantumGate::RZ.is_parametric());

    // ── Name round-trip ──
    report.check(
        "Parse 'quantum.h' → H",
        QuantumGate::from_name("quantum.h") == Some(QuantumGate::H),
    );
    report.check(
        "Parse 'quantum.cx' → CX",
        QuantumGate::from_name("quantum.cx") == Some(QuantumGate::CX),
    );

    // ── Hardware native gate sets ──
    let ibm = QuantumGate::native_basis(Provider::IbmEagle);
    println!(
        "    IBM Eagle native: {:?}",
        ibm.iter().map(|g| g.op_name()).collect::<Vec<_>>()
    );
    report.check("IBM Eagle has native gates", !ibm.is_empty());

    let ionq = QuantumGate::native_basis(Provider::IonQ);
    println!(
        "    IonQ native: {:?}",
        ionq.iter().map(|g| g.op_name()).collect::<Vec<_>>()
    );
    report.check("IonQ has native gates", !ionq.is_empty());

    report.check("CCX is 3-qubit", QuantumGate::CCX.num_qubits() == 3);
    report.check("Measure is measurement", QuantumGate::Measure.is_measurement());
}

// ────────────────────────────────────────────────────────────────────────────
// Hybrid encoding and gradient methods (lift-hybrid)
// ────────────────────────────────────────────────────────────────────────────

/// Test encoding strategies, gradient methods, and joint gradient evaluation
/// counts for the hybrid classical-quantum pipeline.
pub fn test_hybrid_encoding(report: &mut TestReport) {
    use lift_hybrid::encoding::{EncodingConfig, EncodingStrategy};
    use lift_hybrid::gradient::{GradientMethod, JointGradientConfig};

    // ── Encoding strategies ──
    let angle = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);
    println!(
        "    AngleEncoding(4): qubits={}, depth={}",
        angle.num_qubits,
        angle.strategy.circuit_depth(4)
    );
    report.check("AngleEncoding: 4 qubits", angle.num_qubits == 4);
    report.check("AngleEncoding: depth 1", angle.strategy.circuit_depth(4) == 1);

    let amp = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 16);
    println!(
        "    AmplitudeEncoding(16): qubits={}, depth={}",
        amp.num_qubits,
        amp.strategy.circuit_depth(16)
    );
    report.check("AmplitudeEncoding: 4 qubits for 16 features", amp.num_qubits == 4);

    let iqp = EncodingConfig::new(EncodingStrategy::IQPEncoding, 8);
    println!(
        "    IQPEncoding(8): qubits={}, depth={}",
        iqp.num_qubits,
        iqp.strategy.circuit_depth(8)
    );
    report.check("IQPEncoding: 8 qubits", iqp.num_qubits == 8);

    // ── Gradient methods ──
    let num_params = 8; // VQC has 4 RY + 4 RZ = 8 parametrised gates

    let ps = GradientMethod::ParameterShift;
    println!(
        "    ParameterShift({} params): evals={}, exact={}",
        num_params,
        ps.circuit_evaluations(num_params),
        ps.is_exact()
    );
    report.check("ParamShift: 2N evaluations", ps.circuit_evaluations(num_params) == 2 * num_params);
    report.check("ParamShift: is exact", ps.is_exact());

    report.check("SPSA: 2 evaluations", GradientMethod::SPSA.circuit_evaluations(num_params) == 2);
    report.check("SPSA: not exact", !GradientMethod::SPSA.is_exact());

    report.check("Adjoint: 1 evaluation", GradientMethod::Adjoint.circuit_evaluations(num_params) == 1);
    report.check("Adjoint: is exact", GradientMethod::Adjoint.is_exact());

    // ── Joint gradient (classical backprop + quantum parameter shift) ──
    let joint = JointGradientConfig {
        classical_method: GradientMethod::Backprop,
        quantum_method: GradientMethod::ParameterShift,
        num_classical_params: 500,
        num_quantum_params: 8,
    };
    let total = joint.total_evaluations();
    println!("    Joint gradient: {} total evals (1 backprop + 16 param shift)", total);
    report.check("Joint gradient total = 17", total == 17);

    // ── Hybrid op round-trip ──
    report.check(
        "Parse 'hybrid.encode'",
        lift_hybrid::HybridOp::from_name("hybrid.encode").is_some(),
    );
    report.check(
        "'hybrid.parameter_shift' is gradient",
        lift_hybrid::HybridOp::from_name("hybrid.parameter_shift")
            .map(|o| o.is_gradient())
            .unwrap_or(false),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 2 tests and return the analysis reports for use by later steps.
pub fn run(
    cnn_ctx: &Context,
    vqc_ctx: &Context,
    report: &mut TestReport,
) -> (AnalysisReport, QuantumAnalysis) {
    let cnn_report = analyse_cnn(cnn_ctx, report);
    let vqc_analysis = analyse_vqc(vqc_ctx, report);
    test_shape_inference(report);
    test_quantum_gates(report);
    test_hybrid_encoding(report);
    (cnn_report, vqc_analysis)
}
