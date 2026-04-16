// ============================================================================
// LIFT Integration Test — Hybrid AI+Quantum Medical Image Classification
// ============================================================================
//
// This program rigorously tests every major LIFT crate on a real-world problem:
// classifying chest X-rays (pneumonia vs normal) using a hybrid CNN + VQC model.
//
// Steps:
//   1. Parse .lif files (lift-ast)
//   2. Build IR programmatically (lift-core)
//   3. Verify IR correctness (lift-core::verifier)
//   4. Analyse resources: FLOPs, memory, gate count (lift-sim)
//   5. Run optimisation passes (lift-opt)
//   6. Predict GPU performance via roofline model (lift-predict)
//   7. Predict quantum fidelity and shot count (lift-predict)
//   8. Model quantum noise (lift-quantum::noise)
//   9. Evaluate device topology and routing cost (lift-quantum::topology)
//  10. Estimate energy consumption and CO₂ (lift-sim::cost)
//  11. Enforce budget constraints (lift-sim::cost)
//  12. Evaluate hybrid encoding strategies and gradients (lift-hybrid)
//  13. Parse .lith configuration (lift-config)
//  14. Export to LLVM IR and OpenQASM (lift-export)
//  15. Print human-readable IR (lift-core::printer)
// ============================================================================

use lift_core::{Context, Attributes, Location};
use lift_core::types::{Dimension, DataType, MemoryLayout, TensorTypeInfo};
use lift_core::pass::PassManager;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LIFT Integration Test — Hybrid Medical Image Classifier    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut passed = 0u32;
    let mut failed = 0u32;

    // ── Step 1: Parse .lif files with lift-ast ──────────────────────────
    print_step(1, "Parse .lif files (lift-ast)");

    let cnn_ok = test_parse_lif("examples/cnn_encoder.lif");
    check(&mut passed, &mut failed, "Parse CNN encoder .lif", cnn_ok.is_some());

    let vqc_ok = test_parse_lif("examples/quantum_vqc.lif");
    check(&mut passed, &mut failed, "Parse quantum VQC .lif", vqc_ok.is_some());

    // ── Step 2: Build IR programmatically (lift-core) ───────────────────
    print_step(2, "Build hybrid IR programmatically (lift-core)");

    let mut ctx = Context::new();
    let cnn_block = build_cnn_encoder_ir(&mut ctx);
    check(&mut passed, &mut failed, "Build CNN encoder IR", cnn_block.is_some());

    let mut qctx = Context::new();
    let vqc_block = build_vqc_circuit_ir(&mut qctx);
    check(&mut passed, &mut failed, "Build VQC circuit IR", vqc_block.is_some());

    // ── Step 3: Verify IR (lift-core::verifier) ─────────────────────────
    print_step(3, "Verify IR correctness (SSA, types, linearity)");

    let cnn_verify = lift_core::verifier::verify(&ctx);
    match &cnn_verify {
        Ok(()) => println!("    CNN IR: PASSED (SSA + types OK)"),
        Err(errs) => {
            println!("    CNN IR: {} error(s)", errs.len());
            for e in errs { println!("      - {}", e); }
        }
    }
    check(&mut passed, &mut failed, "Verify CNN IR", cnn_verify.is_ok());

    let vqc_verify = lift_core::verifier::verify(&qctx);
    match &vqc_verify {
        Ok(()) => println!("    VQC IR: PASSED (SSA + linearity OK)"),
        Err(errs) => {
            println!("    VQC IR: {} error(s)", errs.len());
            for e in errs { println!("      - {}", e); }
        }
    }
    check(&mut passed, &mut failed, "Verify VQC IR", vqc_verify.is_ok());

    // ── Step 4: Print IR (lift-core::printer) ───────────────────────────
    print_step(4, "Print human-readable IR (lift-core::printer)");

    let cnn_ir = lift_core::printer::print_ir(&ctx);
    println!("    CNN IR length: {} chars", cnn_ir.len());
    check(&mut passed, &mut failed, "Print CNN IR", !cnn_ir.is_empty());

    let vqc_ir = lift_core::printer::print_ir(&qctx);
    println!("    VQC IR length: {} chars", vqc_ir.len());
    check(&mut passed, &mut failed, "Print VQC IR", !vqc_ir.is_empty());

    // ── Step 5: Analyse resources (lift-sim) ────────────────────────────
    print_step(5, "Analyse resources: FLOPs, memory, gates (lift-sim)");

    let cnn_report = lift_sim::analyze_module(&ctx);
    println!("    CNN Analysis:");
    println!("      Total ops:    {}", cnn_report.num_ops);
    println!("      Tensor ops:   {}", cnn_report.num_tensor_ops);
    println!("      Total FLOPs:  {}", format_flops(cnn_report.total_flops));
    println!("      Total memory: {}", format_bytes(cnn_report.total_memory_bytes));
    println!("      Peak memory:  {}", format_bytes(cnn_report.peak_memory_bytes));
    for (op, count) in &cnn_report.op_breakdown {
        println!("        {}: {}", op, count);
    }
    check(&mut passed, &mut failed, "CNN FLOPs > 0", cnn_report.total_flops > 0);
    check(&mut passed, &mut failed, "CNN memory > 0", cnn_report.total_memory_bytes > 0);

    let vqc_quantum = lift_sim::analyze_quantum_ops(&qctx);
    println!("    VQC Quantum Analysis:");
    println!("      Qubits used:    {}", vqc_quantum.num_qubits_used);
    println!("      Gate count:     {}", vqc_quantum.gate_count);
    println!("      1Q gates:       {}", vqc_quantum.one_qubit_gates);
    println!("      2Q gates:       {}", vqc_quantum.two_qubit_gates);
    println!("      Est. fidelity:  {:.6}", vqc_quantum.estimated_fidelity);
    check(&mut passed, &mut failed, "VQC gate count > 0", vqc_quantum.gate_count > 0);
    check(&mut passed, &mut failed, "VQC has 1Q gates", vqc_quantum.one_qubit_gates > 0);
    check(&mut passed, &mut failed, "VQC has 2Q gates", vqc_quantum.two_qubit_gates > 0);
    check(&mut passed, &mut failed, "VQC fidelity in (0,1]",
        vqc_quantum.estimated_fidelity > 0.0 && vqc_quantum.estimated_fidelity <= 1.0);

    // ── Step 6: Tensor shape inference and FLOPs (lift-tensor) ──────────
    print_step(6, "Shape inference and FLOPs counting (lift-tensor)");

    test_shape_inference(&mut passed, &mut failed);

    // ── Step 7: Quantum gate properties (lift-quantum) ──────────────────
    print_step(7, "Quantum gate properties (lift-quantum)");

    test_quantum_gates(&mut passed, &mut failed);

    // ── Step 8: Hybrid encoding and gradients (lift-hybrid) ─────────────
    print_step(8, "Hybrid encoding strategies and gradient methods (lift-hybrid)");

    test_hybrid_encoding_gradients(&mut passed, &mut failed);

    // ── Step 9: Optimisation passes (lift-opt) ──────────────────────────
    print_step(9, "Optimisation passes (lift-opt)");

    test_optimisation_passes(&mut ctx, &mut qctx, &mut passed, &mut failed);

    // ── Step 10: GPU roofline prediction (lift-predict) ─────────────────
    print_step(10, "GPU roofline performance prediction (lift-predict)");

    test_roofline_prediction(&cnn_report, &mut passed, &mut failed);

    // ── Step 11: Quantum prediction (lift-predict) ──────────────────────
    print_step(11, "Quantum fidelity and shot prediction (lift-predict)");

    test_quantum_prediction(&vqc_quantum, &mut passed, &mut failed);

    // ── Step 12: Noise modelling (lift-quantum::noise) ──────────────────
    print_step(12, "Noise modelling and fidelity tracking (lift-quantum::noise)");

    test_noise_modelling(&mut passed, &mut failed);

    // ── Step 13: Device topology (lift-quantum::topology) ───────────────
    print_step(13, "Device topology and routing cost (lift-quantum::topology)");

    test_device_topology(&mut passed, &mut failed);

    // ── Step 14: Energy and CO2 estimation (lift-sim::cost) ─────────────
    print_step(14, "Energy and carbon estimation (lift-sim::cost)");

    test_energy_estimation(&cnn_report, &mut passed, &mut failed);

    // ── Step 15: Budget enforcement (lift-sim::cost) ────────────────────
    print_step(15, "Budget enforcement — static and reactive (lift-sim::cost)");

    test_budget_enforcement(&cnn_report, &vqc_quantum, &mut passed, &mut failed);

    // ── Step 16: Configuration parsing (lift-config) ────────────────────
    print_step(16, "Configuration parsing .lith (lift-config)");

    test_config_parsing(&mut passed, &mut failed);

    // ── Step 17: Export (lift-export) ───────────────────────────────────
    print_step(17, "Export to LLVM IR and OpenQASM 3.0 (lift-export)");

    test_export(&ctx, &qctx, &mut passed, &mut failed);

    // ── Final Report ────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  FINAL REPORT                                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Passed: {:>3}                                               ║", passed);
    println!("║  Failed: {:>3}                                               ║", failed);
    println!("║  Total:  {:>3}                                               ║", passed + failed);
    println!("╚══════════════════════════════════════════════════════════════╝");

    if failed > 0 {
        std::process::exit(1);
    }
}

// ============================================================================
// Helper: test result tracking
// ============================================================================

fn check(passed: &mut u32, failed: &mut u32, name: &str, ok: bool) {
    if ok {
        println!("    [PASS] {}", name);
        *passed += 1;
    } else {
        println!("    [FAIL] {}", name);
        *failed += 1;
    }
}

fn print_step(n: u32, title: &str) {
    println!();
    println!("── Step {} ── {}", n, title);
}

// ============================================================================
// Step 1: Parse .lif files
// ============================================================================

fn test_parse_lif(path: &str) -> Option<Context> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("    Could not read {}: {}", path, e);
            return None;
        }
    };

    let mut lexer = lift_ast::Lexer::new(&source);
    let tokens = lexer.tokenize().to_vec();
    if !lexer.errors().is_empty() {
        println!("    Lexer errors in {}: {:?}", path, lexer.errors());
        return None;
    }
    println!("    {} tokens from {}", tokens.len(), path);

    let mut parser = lift_ast::Parser::new(tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => {
            println!("    Parse errors in {}: {:?}", path, e);
            return None;
        }
    };

    let mut ctx = Context::new();
    let mut builder = lift_ast::IrBuilder::new();
    match builder.build_program(&mut ctx, &program) {
        Ok(()) => {
            println!("    Built IR: {} ops, {} values, {} blocks",
                ctx.ops.len(), ctx.values.len(), ctx.blocks.len());
            Some(ctx)
        }
        Err(e) => {
            println!("    IR build error in {}: {}", path, e);
            None
        }
    }
}

// ============================================================================
// Step 2: Build CNN encoder IR programmatically
// ============================================================================

fn build_cnn_encoder_ir(ctx: &mut Context) -> Option<()> {
    // Types
    let img_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(1),
             Dimension::Constant(128), Dimension::Constant(128)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let w1_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(32), Dimension::Constant(1),
             Dimension::Constant(3), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let conv1_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(32),
             Dimension::Constant(64), Dimension::Constant(64)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let pool1_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(32),
             Dimension::Constant(32), Dimension::Constant(32)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let w2_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(64), Dimension::Constant(32),
             Dimension::Constant(3), Dimension::Constant(3)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let conv2_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(64),
             Dimension::Constant(16), Dimension::Constant(16)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let gap_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(64)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let wfc_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(64), Dimension::Constant(4)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let bfc_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)],
        DataType::FP32, MemoryLayout::Contiguous,
    );
    let out_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(4)],
        DataType::FP32, MemoryLayout::Contiguous,
    );

    // Block and arguments
    let block = ctx.create_block();
    let img = ctx.create_block_arg(block, img_ty);
    let w1 = ctx.create_block_arg(block, w1_ty);
    let w2 = ctx.create_block_arg(block, w2_ty);
    let wfc = ctx.create_block_arg(block, wfc_ty);
    let bfc = ctx.create_block_arg(block, bfc_ty);

    // conv2d -> relu -> maxpool2d
    let (op1, res1) = ctx.create_op(
        "tensor.conv2d", "tensor", vec![img, w1], vec![conv1_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op1);

    let (op2, res2) = ctx.create_op(
        "tensor.relu", "tensor", vec![res1[0]], vec![conv1_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op2);

    let (op3, res3) = ctx.create_op(
        "tensor.maxpool2d", "tensor", vec![res2[0]], vec![pool1_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op3);

    // conv2d -> relu
    let (op4, res4) = ctx.create_op(
        "tensor.conv2d", "tensor", vec![res3[0], w2], vec![conv2_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op4);

    let (op5, res5) = ctx.create_op(
        "tensor.relu", "tensor", vec![res4[0]], vec![conv2_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op5);

    // global_avgpool
    let (op6, res6) = ctx.create_op(
        "tensor.global_avgpool", "tensor", vec![res5[0]], vec![gap_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op6);

    // matmul + add (linear layer)
    let (op7, res7) = ctx.create_op(
        "tensor.matmul", "tensor", vec![res6[0], wfc], vec![out_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op7);

    let (op8, _res8) = ctx.create_op(
        "tensor.add", "tensor", vec![res7[0], bfc], vec![out_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op8);

    Some(())
}

// ============================================================================
// Step 2b: Build VQC circuit IR programmatically
// ============================================================================

fn build_vqc_circuit_ir(ctx: &mut Context) -> Option<()> {
    let qubit_ty = ctx.make_qubit_type();

    let block = ctx.create_block();
    let q0 = ctx.create_block_arg(block, qubit_ty);
    let q1 = ctx.create_block_arg(block, qubit_ty);
    let q2 = ctx.create_block_arg(block, qubit_ty);
    let q3 = ctx.create_block_arg(block, qubit_ty);

    // Layer 1: RY encoding on all 4 qubits
    let (op_ry0, ry0) = ctx.create_op(
        "quantum.ry", "quantum", vec![q0], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_ry0);

    let (op_ry1, ry1) = ctx.create_op(
        "quantum.ry", "quantum", vec![q1], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_ry1);

    let (op_ry2, ry2) = ctx.create_op(
        "quantum.ry", "quantum", vec![q2], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_ry2);

    let (op_ry3, ry3) = ctx.create_op(
        "quantum.ry", "quantum", vec![q3], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_ry3);

    // Layer 2: Entanglement CX(0,1), CX(2,3)
    let (op_cx01, cx01) = ctx.create_op(
        "quantum.cx", "quantum", vec![ry0[0], ry1[0]], vec![qubit_ty, qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_cx01);

    let (op_cx23, cx23) = ctx.create_op(
        "quantum.cx", "quantum", vec![ry2[0], ry3[0]], vec![qubit_ty, qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_cx23);

    // Layer 3: RZ parametrised rotations
    let (op_rz0, _rz0) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx01[0]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_rz0);

    let (op_rz1, _rz1) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx01[1]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_rz1);

    let (op_rz2, _rz2) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx23[0]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_rz2);

    let (op_rz3, _rz3) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx23[1]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op_rz3);

    Some(())
}

// ============================================================================
// Step 6: Shape inference and FLOPs
// ============================================================================

fn test_shape_inference(passed: &mut u32, failed: &mut u32) {
    use lift_tensor::{TensorOp, ShapeInference};

    // MatMul shape inference
    let a = TensorTypeInfo {
        shape: vec![Dimension::Constant(1), Dimension::Constant(64)],
        dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
    };
    let b = TensorTypeInfo {
        shape: vec![Dimension::Constant(64), Dimension::Constant(4)],
        dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
    };

    let result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]);
    match &result {
        Ok(shapes) => {
            println!("    MatMul [1x64] @ [64x4] -> {:?}",
                shapes[0].shape.iter()
                    .map(|d| format!("{:?}", d)).collect::<Vec<_>>());
        }
        Err(e) => println!("    MatMul shape error: {}", e),
    }
    check(passed, failed, "MatMul shape inference", result.is_ok());

    // MatMul FLOPs: 2*M*N*K = 2*1*4*64 = 512
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
    println!("    MatMul FLOPs: {:?}", flops);
    check(passed, failed, "MatMul FLOPs computation", flops.is_some());

    // Memory estimation
    let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
    println!("    MatMul memory: {:?} bytes", mem);
    check(passed, failed, "MatMul memory estimation", mem.is_some());

    // Conv2d FLOPs
    let img = TensorTypeInfo {
        shape: vec![Dimension::Constant(1), Dimension::Constant(1),
                    Dimension::Constant(128), Dimension::Constant(128)],
        dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
    };
    let kernel = TensorTypeInfo {
        shape: vec![Dimension::Constant(32), Dimension::Constant(1),
                    Dimension::Constant(3), Dimension::Constant(3)],
        dtype: DataType::FP32, layout: MemoryLayout::Contiguous,
    };
    let conv_flops = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&img, &kernel]);
    println!("    Conv2d FLOPs: {:?}", conv_flops);
    check(passed, failed, "Conv2d FLOPs computation", conv_flops.is_some());

    // Zero-FLOP ops (reshape, transpose)
    check(passed, failed, "Reshape is zero-FLOP", TensorOp::Reshape.is_zero_flop());
    check(passed, failed, "MatMul is NOT zero-FLOP", !TensorOp::MatMul.is_zero_flop());

    // Operation categorisation
    check(passed, failed, "ReLU is activation", TensorOp::ReLU.is_activation());
    check(passed, failed, "Attention is attention op", TensorOp::Attention.is_attention());
    check(passed, failed, "Conv2D is convolution", TensorOp::Conv2D.is_convolution());
    check(passed, failed, "FusedMatMulBiasRelu is fused",
        TensorOp::FusedMatMulBiasReLU.is_fused());
}

// ============================================================================
// Step 7: Quantum gate properties
// ============================================================================

fn test_quantum_gates(passed: &mut u32, failed: &mut u32) {
    use lift_quantum::QuantumGate;
    use lift_quantum::gates::Provider;

    // Gate properties
    let h = QuantumGate::H;
    println!("    H gate: name={}, qubits={}, clifford={}, self_inv={}",
        h.op_name(), h.num_qubits(), h.is_clifford(), h.is_self_inverse());
    check(passed, failed, "H is 1-qubit", h.num_qubits() == 1);
    check(passed, failed, "H is Clifford", h.is_clifford());
    check(passed, failed, "H is self-inverse", h.is_self_inverse());

    let cx = QuantumGate::CX;
    println!("    CX gate: name={}, qubits={}, entangling={}",
        cx.op_name(), cx.num_qubits(), cx.is_entangling());
    check(passed, failed, "CX is 2-qubit", cx.num_qubits() == 2);
    check(passed, failed, "CX is entangling", cx.is_entangling());

    let ry = QuantumGate::RY;
    check(passed, failed, "RY is parametric", ry.is_parametric());
    check(passed, failed, "RY is 1-qubit", ry.num_qubits() == 1);

    let rz = QuantumGate::RZ;
    check(passed, failed, "RZ is parametric", rz.is_parametric());

    // Gate name round-trip
    let parsed = QuantumGate::from_name("quantum.h");
    check(passed, failed, "Parse 'quantum.h' -> H", parsed == Some(QuantumGate::H));

    let parsed_cx = QuantumGate::from_name("quantum.cx");
    check(passed, failed, "Parse 'quantum.cx' -> CX", parsed_cx == Some(QuantumGate::CX));

    // Hardware native gate sets
    let ibm = QuantumGate::native_basis(Provider::IbmEagle);
    println!("    IBM Eagle native gates: {:?}",
        ibm.iter().map(|g| g.op_name()).collect::<Vec<_>>());
    check(passed, failed, "IBM Eagle has native gates", !ibm.is_empty());

    let ionq = QuantumGate::native_basis(Provider::IonQ);
    println!("    IonQ native gates: {:?}",
        ionq.iter().map(|g| g.op_name()).collect::<Vec<_>>());
    check(passed, failed, "IonQ has native gates", !ionq.is_empty());

    // CCX (Toffoli) is 3-qubit
    check(passed, failed, "CCX is 3-qubit", QuantumGate::CCX.num_qubits() == 3);

    // Measurement gate
    check(passed, failed, "Measure is measurement", QuantumGate::Measure.is_measurement());
}

// ============================================================================
// Step 8: Hybrid encoding and gradient methods
// ============================================================================

fn test_hybrid_encoding_gradients(passed: &mut u32, failed: &mut u32) {
    use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};
    use lift_hybrid::gradient::{GradientMethod, JointGradientConfig};

    // Angle encoding: 4 features → 4 qubits, depth 1
    let angle = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);
    println!("    AngleEncoding(4): qubits={}, depth={}",
        angle.num_qubits, angle.strategy.circuit_depth(4));
    check(passed, failed, "AngleEncoding: 4 qubits", angle.num_qubits == 4);
    check(passed, failed, "AngleEncoding: depth 1",
        angle.strategy.circuit_depth(4) == 1);

    // Amplitude encoding: 16 features → 4 qubits
    let amp = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 16);
    println!("    AmplitudeEncoding(16): qubits={}, depth={}",
        amp.num_qubits, amp.strategy.circuit_depth(16));
    check(passed, failed, "AmplitudeEncoding: 4 qubits for 16 features",
        amp.num_qubits == 4);

    // IQP encoding: 8 features → 8 qubits
    let iqp = EncodingConfig::new(EncodingStrategy::IQPEncoding, 8);
    println!("    IQPEncoding(8): qubits={}, depth={}",
        iqp.num_qubits, iqp.strategy.circuit_depth(8));
    check(passed, failed, "IQPEncoding: 8 qubits", iqp.num_qubits == 8);

    // Parameter shift: exact, 2N evaluations
    let ps = GradientMethod::ParameterShift;
    let num_params = 8; // our VQC has 8 parametrised gates
    println!("    ParameterShift({} params): evals={}, exact={}",
        num_params, ps.circuit_evaluations(num_params), ps.is_exact());
    check(passed, failed, "ParamShift: 2N evaluations",
        ps.circuit_evaluations(num_params) == 2 * num_params);
    check(passed, failed, "ParamShift: is exact", ps.is_exact());

    // SPSA: 2 evaluations regardless of params
    let spsa = GradientMethod::SPSA;
    check(passed, failed, "SPSA: 2 evaluations",
        spsa.circuit_evaluations(num_params) == 2);
    check(passed, failed, "SPSA: not exact", !spsa.is_exact());

    // Adjoint: 1 evaluation
    let adj = GradientMethod::Adjoint;
    check(passed, failed, "Adjoint: 1 evaluation",
        adj.circuit_evaluations(num_params) == 1);
    check(passed, failed, "Adjoint: is exact", adj.is_exact());

    // Joint gradient: classical backprop + quantum parameter shift
    let joint = JointGradientConfig {
        classical_method: GradientMethod::Backprop,
        quantum_method: GradientMethod::ParameterShift,
        num_classical_params: 500,
        num_quantum_params: 8,
    };
    let total = joint.total_evaluations();
    println!("    Joint gradient: {} classical + {} quantum = {} total evals",
        joint.num_classical_params, joint.num_quantum_params, total);
    check(passed, failed, "Joint gradient total = 1 + 16 = 17", total == 17);

    // Hybrid op round-trip
    let encode_op = lift_hybrid::HybridOp::from_name("hybrid.encode");
    check(passed, failed, "Parse 'hybrid.encode'", encode_op.is_some());

    let ps_op = lift_hybrid::HybridOp::from_name("hybrid.parameter_shift");
    check(passed, failed, "'hybrid.parameter_shift' is gradient",
        ps_op.map(|o| o.is_gradient()).unwrap_or(false));
}

// ============================================================================
// Step 9: Optimisation passes
// ============================================================================

fn test_optimisation_passes(
    ctx: &mut Context, qctx: &mut Context,
    passed: &mut u32, failed: &mut u32,
) {
    // CNN: canonicalize + tensor fusion + DCE
    let ops_before = ctx.ops.len();
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::TensorFusion));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

    let results = pm.run_all(ctx);
    println!("    CNN passes:");
    for (name, result) in &results {
        let status = match result {
            lift_core::PassResult::Changed => "changed",
            lift_core::PassResult::Unchanged => "unchanged",
            lift_core::PassResult::RolledBack => "rolled back",
            lift_core::PassResult::Error(e) => { println!("      {} -> error: {}", name, e); "error" }
        };
        println!("      {} -> {}", name, status);
    }
    check(passed, failed, "CNN passes ran without panic", true);
    println!("    CNN ops: {} -> {}", ops_before, ctx.ops.len());

    // Quantum: gate cancellation + rotation merge
    let qops_before = qctx.ops.len();
    let mut qpm = PassManager::new();
    qpm.add_pass(Box::new(lift_opt::GateCancellation));
    qpm.add_pass(Box::new(lift_opt::RotationMerge));

    let qresults = qpm.run_all(qctx);
    println!("    VQC passes:");
    for (name, result) in &qresults {
        let status = match result {
            lift_core::PassResult::Changed => "changed",
            lift_core::PassResult::Unchanged => "unchanged",
            lift_core::PassResult::RolledBack => "rolled back",
            lift_core::PassResult::Error(e) => { println!("      {} -> error: {}", name, e); "error" }
        };
        println!("      {} -> {}", name, status);
    }
    check(passed, failed, "VQC passes ran without panic", true);
    println!("    VQC ops: {} -> {}", qops_before, qctx.ops.len());
}

// ============================================================================
// Step 10: GPU roofline prediction
// ============================================================================

fn test_roofline_prediction(
    report: &lift_sim::AnalysisReport,
    passed: &mut u32, failed: &mut u32,
) {
    let a100 = lift_sim::cost::CostModel::a100();
    let h100 = lift_sim::cost::CostModel::h100();

    let pred_a100 = lift_predict::predict_performance(report, &a100);
    let pred_h100 = lift_predict::predict_performance(report, &h100);

    println!("    A100 prediction:");
    println!("      Compute time:  {:.6} ms", pred_a100.compute_time_ms);
    println!("      Memory time:   {:.6} ms", pred_a100.memory_time_ms);
    println!("      Predicted:     {:.6} ms", pred_a100.predicted_time_ms);
    println!("      Arith intens:  {:.2} FLOP/byte", pred_a100.arithmetic_intensity);
    println!("      Bottleneck:    {}", pred_a100.bottleneck);

    println!("    H100 prediction:");
    println!("      Predicted:     {:.6} ms", pred_h100.predicted_time_ms);
    if pred_h100.predicted_time_ms > 0.0 {
        println!("      Speedup vs A100: {:.2}x",
            pred_a100.predicted_time_ms / pred_h100.predicted_time_ms);
    }

    check(passed, failed, "A100 predicted time >= 0", pred_a100.predicted_time_ms >= 0.0);
    check(passed, failed, "H100 predicted time >= 0", pred_h100.predicted_time_ms >= 0.0);
    check(passed, failed, "Bottleneck is 'compute' or 'memory'",
        pred_a100.bottleneck == "compute" || pred_a100.bottleneck == "memory");

    // Memory fit check
    let fits = a100.fits_in_memory(report.total_memory_bytes);
    let gpus = a100.num_gpus_needed(report.total_memory_bytes);
    println!("    Fits in 1 A100: {} ({} GPU(s) needed)", fits, gpus);
    check(passed, failed, "CNN fits in 1 GPU", fits);
}

// ============================================================================
// Step 11: Quantum prediction
// ============================================================================

fn test_quantum_prediction(
    analysis: &lift_sim::QuantumAnalysis,
    passed: &mut u32, failed: &mut u32,
) {
    let sc = lift_sim::cost::QuantumCostModel::superconducting_default();
    let ti = lift_sim::cost::QuantumCostModel::trapped_ion_default();

    let pred_sc = lift_predict::predict_quantum(analysis, &sc, 0.01);
    let pred_ti = lift_predict::predict_quantum(analysis, &ti, 0.01);

    println!("    Superconducting (IBM-like):");
    println!("      Fidelity:  {:.6}", pred_sc.estimated_fidelity);
    println!("      Circuit:   {:.4} us", pred_sc.circuit_time_us);
    println!("      Shots:     {} (for 1% precision)", pred_sc.num_shots_for_precision);
    println!("      Total:     {:.4} ms", pred_sc.total_execution_time_ms);

    println!("    Trapped-ion (IonQ-like):");
    println!("      Fidelity:  {:.6}", pred_ti.estimated_fidelity);
    println!("      Circuit:   {:.4} us", pred_ti.circuit_time_us);
    println!("      Shots:     {}", pred_ti.num_shots_for_precision);
    println!("      Total:     {:.4} ms", pred_ti.total_execution_time_ms);

    check(passed, failed, "SC fidelity in (0,1]",
        pred_sc.estimated_fidelity > 0.0 && pred_sc.estimated_fidelity <= 1.0);
    check(passed, failed, "TI fidelity in (0,1]",
        pred_ti.estimated_fidelity > 0.0 && pred_ti.estimated_fidelity <= 1.0);
    check(passed, failed, "TI fidelity > SC fidelity",
        pred_ti.estimated_fidelity > pred_sc.estimated_fidelity);
    check(passed, failed, "Shot count > 0", pred_sc.num_shots_for_precision > 0);
}

// ============================================================================
// Step 12: Noise modelling
// ============================================================================

fn test_noise_modelling(passed: &mut u32, failed: &mut u32) {
    use lift_quantum::noise::{NoiseModel, GateNoise, CircuitNoise};

    // Individual noise models
    let depol = NoiseModel::Depolarizing { p: 0.001 };
    let fid_depol = depol.fidelity();
    println!("    Depolarizing(p=0.001): fidelity={:.6}", fid_depol);
    check(passed, failed, "Depolarizing fidelity ~0.999",
        (fid_depol - 0.999).abs() < 0.001);

    let thermal = NoiseModel::ThermalRelaxation {
        t1_us: 100.0, t2_us: 80.0, gate_time_us: 0.3,
    };
    let fid_therm = thermal.fidelity();
    println!("    Thermal(T1=100, T2=80, t=0.3): fidelity={:.6}", fid_therm);
    check(passed, failed, "Thermal fidelity in (0,1]",
        fid_therm > 0.0 && fid_therm <= 1.0);

    // Composed noise
    let composed = depol.compose(&thermal);
    let fid_comp = composed.fidelity();
    println!("    Composed fidelity: {:.6}", fid_comp);
    check(passed, failed, "Composed fidelity <= min(individual)",
        fid_comp <= fid_depol && fid_comp <= fid_therm);

    // Circuit-level noise tracking for our VQC (4x RY + 2x CX + 4x RZ)
    let mut circuit = CircuitNoise::new();
    let g1q = GateNoise::with_depolarizing(0.999, 0.02);
    let g2q = GateNoise::with_depolarizing(0.99, 0.3);

    // 4x RY (1Q)
    for _ in 0..4 { circuit.add_gate(&g1q, false); }
    let after_ry = circuit.total_fidelity;
    println!("    After 4x RY: fidelity={:.6}", after_ry);

    // 2x CX (2Q)
    for _ in 0..2 { circuit.add_gate(&g2q, true); }
    let after_cx = circuit.total_fidelity;
    println!("    After 2x CX: fidelity={:.6}", after_cx);

    // 4x RZ (1Q)
    for _ in 0..4 { circuit.add_gate(&g1q, false); }
    let final_fid = circuit.total_fidelity;
    println!("    After 4x RZ: fidelity={:.6} (final)", final_fid);

    println!("    Circuit: {} gates, {} 2Q gates",
        circuit.gate_count, circuit.two_qubit_count);

    check(passed, failed, "Fidelity degrades: after_CX < after_RY",
        after_cx < after_ry);
    check(passed, failed, "Total gates = 10", circuit.gate_count == 10);
    check(passed, failed, "2Q gates = 2", circuit.two_qubit_count == 2);
    check(passed, failed, "Meets 90% threshold", circuit.meets_threshold(0.90));
}

// ============================================================================
// Step 13: Device topology
// ============================================================================

fn test_device_topology(passed: &mut u32, failed: &mut u32) {
    use lift_quantum::DeviceTopology;

    // Grid topology (like Google Sycamore)
    let grid = DeviceTopology::grid(2, 2); // 4 qubits for our VQC
    println!("    Grid 2x2: {} qubits, {} edges",
        grid.num_qubits, grid.edges.len());
    check(passed, failed, "Grid has 4 qubits", grid.num_qubits == 4);

    // Connectivity checks
    let conn_01 = grid.are_connected(0, 1);
    println!("    0-1 connected: {}", conn_01);
    check(passed, failed, "Grid: 0-1 connected", conn_01);

    // Shortest path
    if let Some(path) = grid.shortest_path(0, 3) {
        println!("    Path 0->3: {:?} ({} SWAPs)", path, path.len().saturating_sub(2));
        check(passed, failed, "Path 0->3 found", true);
    } else {
        check(passed, failed, "Path 0->3 found", false);
    }

    // Neighbours
    let neighbors = grid.neighbors(0);
    println!("    Neighbours of qubit 0: {:?}", neighbors);
    check(passed, failed, "Qubit 0 has neighbours", !neighbors.is_empty());

    // Heavy-hex (IBM)
    let hh = DeviceTopology::heavy_hex(127);
    println!("    Heavy-hex: {} qubits, {} edges, diameter {}",
        hh.num_qubits, hh.edges.len(), hh.diameter());
    check(passed, failed, "Heavy-hex has 127 qubits", hh.num_qubits == 127);

    // All-to-all (trapped-ion)
    let ata = DeviceTopology::all_to_all(4);
    println!("    All-to-all(4): {} edges, avg connectivity {:.2}",
        ata.edges.len(), ata.avg_connectivity());
    check(passed, failed, "All-to-all(4) has 6 edges", ata.edges.len() == 6);

    // SWAP distance in all-to-all should be 0 (direct connection)
    let swap_dist = ata.swap_distance(0, 3);
    println!("    All-to-all SWAP distance 0->3: {:?}", swap_dist);
    check(passed, failed, "All-to-all: 0 SWAPs needed",
        swap_dist == Some(0));

    // Linear chain
    let linear = DeviceTopology::linear(4);
    let swap_03 = linear.swap_distance(0, 3);
    println!("    Linear(4) SWAP distance 0->3: {:?}", swap_03);
    check(passed, failed, "Linear: 0->3 needs 2 SWAPs", swap_03 == Some(2));
}

// ============================================================================
// Step 14: Energy and CO2 estimation
// ============================================================================

fn test_energy_estimation(
    report: &lift_sim::AnalysisReport,
    passed: &mut u32, failed: &mut u32,
) {
    let a100_cost = lift_sim::cost::CostModel::a100();
    let a100_pred = lift_predict::predict_performance(report, &a100_cost);
    let energy = lift_sim::cost::EnergyModel::a100();

    // Single inference
    let joules = energy.energy_joules(a100_pred.predicted_time_ms, 1);
    let kwh = energy.energy_kwh(a100_pred.predicted_time_ms, 1);
    let co2 = energy.carbon_grams(a100_pred.predicted_time_ms, 1);

    println!("    Single inference on A100:");
    println!("      Energy: {:.6} J ({:.10} kWh)", joules, kwh);
    println!("      CO2: {:.8} g", co2);
    check(passed, failed, "Energy >= 0", joules >= 0.0);
    check(passed, failed, "CO2 >= 0", co2 >= 0.0);

    // Training simulation: 8 GPUs, 24h
    let train_ms = 24.0 * 3600.0 * 1000.0;
    let train_kwh = energy.energy_kwh(train_ms, 8);
    let train_co2_kg = energy.carbon_grams(train_ms, 8) / 1000.0;
    println!("    Training (8x A100, 24h):");
    println!("      Energy: {:.2} kWh", train_kwh);
    println!("      CO2: {:.2} kg", train_co2_kg);
    check(passed, failed, "Training energy > 0", train_kwh > 0.0);

    // Quantum energy
    let q_joules = energy.quantum_energy_joules(100.0, 4);
    println!("    Quantum (100us, 4 qubits): {:.6} J", q_joules);
    check(passed, failed, "Quantum energy > 0", q_joules > 0.0);
}

// ============================================================================
// Step 15: Budget enforcement
// ============================================================================

fn test_budget_enforcement(
    report: &lift_sim::AnalysisReport,
    quantum: &lift_sim::QuantumAnalysis,
    passed: &mut u32, failed: &mut u32,
) {
    use lift_sim::cost::{Budget, ReactiveBudget};

    // Static budget: generous (should pass)
    let generous = Budget {
        max_flops: Some(100_000_000_000),
        max_memory_bytes: Some(80_000_000_000),
        max_time_ms: Some(1000.0),
        min_fidelity: Some(0.50),
        max_circuit_depth: None,
    };
    check(passed, failed, "Generous FLOP budget OK",
        generous.check_flops(report.total_flops).is_ok());
    check(passed, failed, "Generous memory budget OK",
        generous.check_memory(report.total_memory_bytes).is_ok());
    check(passed, failed, "Generous fidelity OK",
        generous.check_fidelity(quantum.estimated_fidelity).is_ok());

    // Tight budget: should fail
    let tight = Budget {
        max_flops: Some(100),
        max_memory_bytes: Some(100),
        max_time_ms: None,
        min_fidelity: Some(0.9999),
        max_circuit_depth: None,
    };
    check(passed, failed, "Tight FLOP budget FAILS",
        tight.check_flops(report.total_flops).is_err());
    check(passed, failed, "Tight memory budget FAILS",
        tight.check_memory(report.total_memory_bytes).is_err());

    // Reactive budget: VQE-style iteration
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: Some(1000.0),
        min_fidelity: Some(0.80),
        max_circuit_depth: None,
    };
    let mut tracker = ReactiveBudget::new(budget);

    let mut iters_completed = 0;
    for i in 0..200 {
        tracker.consume(0, 0, 10.0, 0.999);
        if tracker.check_remaining().is_err() {
            println!("    Reactive budget exhausted at iteration {}", i);
            break;
        }
        iters_completed = i + 1;
    }
    println!("    Completed {} iterations, elapsed={:.0}ms, fidelity={:.6}",
        iters_completed, tracker.elapsed_ms, tracker.current_fidelity);

    check(passed, failed, "Reactive: stopped before 200 iters", iters_completed < 200);
    check(passed, failed, "Reactive: elapsed > 0", tracker.elapsed_ms > 0.0);

    let util = tracker.utilisation();
    if let Some(time_ratio) = util.time_ratio {
        println!("    Time utilisation: {:.1}%", time_ratio * 100.0);
        check(passed, failed, "Time utilisation near or past 100%", time_ratio >= 0.9);
    }
}

// ============================================================================
// Step 16: Config parsing
// ============================================================================

fn test_config_parsing(passed: &mut u32, failed: &mut u32) {
    // Parse .lith file
    let lith_src = match std::fs::read_to_string("examples/hybrid_config.lith") {
        Ok(s) => s,
        Err(e) => {
            println!("    Could not read .lith file: {}", e);
            check(passed, failed, "Read .lith file", false);
            return;
        }
    };
    check(passed, failed, "Read .lith file", true);

    let parser = lift_config::ConfigParser::new();
    let config = parser.parse(&lith_src);
    match &config {
        Ok(c) => {
            println!("    target.backend = {}", c.target.backend);
            println!("    target.device = {:?}", c.target.device);
            println!("    optimisation.level = {:?}", c.optimisation.level);
            println!("    budget.max_flops = {:?}", c.budget.max_flops);
            println!("    budget.min_fidelity = {:?}", c.budget.min_fidelity);
            if let Some(q) = &c.quantum {
                println!("    quantum.topology = {}", q.topology);
                println!("    quantum.num_qubits = {}", q.num_qubits);
                println!("    quantum.shots = {:?}", q.shots);
            }
        }
        Err(e) => println!("    Parse error: {}", e),
    }
    check(passed, failed, "Parse .lith config", config.is_ok());

    if let Ok(c) = &config {
        check(passed, failed, "Backend = llvm", c.target.backend == "llvm");
        check(passed, failed, "Opt level = O3",
            c.optimisation.level == lift_config::OptLevel::O3);
        check(passed, failed, "Quantum section present", c.quantum.is_some());
        if let Some(q) = &c.quantum {
            check(passed, failed, "Quantum topology = grid", q.topology == "grid");
            check(passed, failed, "Quantum qubits = 4", q.num_qubits == 4);
        }
        check(passed, failed, "Budget min_fidelity = 0.90",
            c.budget.min_fidelity == Some(0.90));
    }

    // Test default config
    let default = lift_config::LithConfig::default();
    check(passed, failed, "Default backend = llvm", default.target.backend == "llvm");
    check(passed, failed, "Default opt level = O2",
        default.optimisation.level == lift_config::OptLevel::O2);
}

// ============================================================================
// Step 17: Export
// ============================================================================

fn test_export(ctx: &Context, qctx: &Context, passed: &mut u32, failed: &mut u32) {
    // LLVM export (tensor/classical)
    let llvm_exporter = lift_export::LlvmExporter::new();
    let llvm_result = llvm_exporter.export(ctx);
    match &llvm_result {
        Ok(ir) => println!("    LLVM IR: {} bytes", ir.len()),
        Err(e) => println!("    LLVM export error: {}", e),
    }
    check(passed, failed, "LLVM export succeeds", llvm_result.is_ok());
    if let Ok(ir) = &llvm_result {
        check(passed, failed, "LLVM IR non-empty", !ir.is_empty());
    }

    // QASM export (quantum)
    let qasm_exporter = lift_export::QasmExporter::new();
    let qasm_result = qasm_exporter.export(qctx);
    match &qasm_result {
        Ok(qasm) => println!("    OpenQASM: {} bytes", qasm.len()),
        Err(e) => println!("    QASM export error: {}", e),
    }
    check(passed, failed, "QASM export succeeds", qasm_result.is_ok());
    if let Ok(qasm) = &qasm_result {
        check(passed, failed, "QASM output non-empty", !qasm.is_empty());
    }
}

// ============================================================================
// Formatting helpers
// ============================================================================

fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 { format!("{:.2} TFLOP", flops as f64 / 1e12) }
    else if flops >= 1_000_000_000 { format!("{:.2} GFLOP", flops as f64 / 1e9) }
    else if flops >= 1_000_000 { format!("{:.2} MFLOP", flops as f64 / 1e6) }
    else if flops >= 1_000 { format!("{:.2} KFLOP", flops as f64 / 1e3) }
    else { format!("{} FLOP", flops) }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 { format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0) }
    else if bytes >= 1_048_576 { format!("{:.2} MiB", bytes as f64 / 1_048_576.0) }
    else if bytes >= 1_024 { format!("{:.2} KiB", bytes as f64 / 1_024.0) }
    else { format!("{} B", bytes) }
}
