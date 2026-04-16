// ============================================================================
// ir_builder.rs — Programmatic IR Construction
// ============================================================================
//
// Builds LIFT IR programmatically for the two components of the hybrid model:
//
//   1. CNN Encoder (tensor dialect)
//      conv2d → relu → maxpool2d → conv2d → relu → global_avgpool → matmul → add
//      Input:  tensor<1×1×128×128×f32>  (grayscale chest X-ray)
//      Output: tensor<1×4×f32>          (4-dimensional feature vector)
//
//   2. VQC Circuit (quantum dialect)
//      4× RY (angle encoding) → 2× CX (entanglement) → 4× RZ (parametrised)
//      Input:  4 qubits
//      Output: 4 qubits (to be measured)
//
// Both functions return a fully constructed `Context` ready for verification,
// analysis, optimisation, prediction, and export.
//
// ============================================================================

use lift_core::types::{DataType, Dimension, MemoryLayout};
use lift_core::{Attributes, Context, Location};

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// CNN Encoder — classical tensor IR
// ────────────────────────────────────────────────────────────────────────────

/// Build the CNN encoder IR and return the populated `Context`.
///
/// Architecture:
/// ```text
/// img[1,1,128,128] → conv2d(32 filters) → relu → maxpool2d
///                   → conv2d(64 filters) → relu → global_avgpool
///                   → matmul(64→4) → add(bias) → features[1,4]
/// ```
pub fn build_cnn_context() -> Context {
    let mut ctx = Context::new();

    // ── Tensor types ──
    let img_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(1),
            Dimension::Constant(1),
            Dimension::Constant(128),
            Dimension::Constant(128),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let w1_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(32),
            Dimension::Constant(1),
            Dimension::Constant(3),
            Dimension::Constant(3),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let conv1_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(1),
            Dimension::Constant(32),
            Dimension::Constant(64),
            Dimension::Constant(64),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let pool1_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(1),
            Dimension::Constant(32),
            Dimension::Constant(32),
            Dimension::Constant(32),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let w2_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(64),
            Dimension::Constant(32),
            Dimension::Constant(3),
            Dimension::Constant(3),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let conv2_ty = ctx.make_tensor_type(
        vec![
            Dimension::Constant(1),
            Dimension::Constant(64),
            Dimension::Constant(16),
            Dimension::Constant(16),
        ],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let gap_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(64)],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let wfc_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(64), Dimension::Constant(4)],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let bfc_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );
    let out_ty = ctx.make_tensor_type(
        vec![Dimension::Constant(1), Dimension::Constant(4)],
        DataType::FP32,
        MemoryLayout::Contiguous,
    );

    // ── Block and arguments ──
    let block = ctx.create_block();
    let img = ctx.create_block_arg(block, img_ty);
    let w1 = ctx.create_block_arg(block, w1_ty);
    let w2 = ctx.create_block_arg(block, w2_ty);
    let wfc = ctx.create_block_arg(block, wfc_ty);
    let bfc = ctx.create_block_arg(block, bfc_ty);

    // ── Layer 1: conv2d → relu → maxpool2d ──
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

    // ── Layer 2: conv2d → relu ──
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

    // ── Global average pooling ──
    let (op6, res6) = ctx.create_op(
        "tensor.global_avgpool", "tensor", vec![res5[0]], vec![gap_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op6);

    // ── Fully-connected: matmul + bias ──
    let (op7, res7) = ctx.create_op(
        "tensor.matmul", "tensor", vec![res6[0], wfc], vec![out_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op7);

    let (op8, _) = ctx.create_op(
        "tensor.add", "tensor", vec![res7[0], bfc], vec![out_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op8);

    ctx
}

// ────────────────────────────────────────────────────────────────────────────
// VQC Circuit — quantum IR
// ────────────────────────────────────────────────────────────────────────────

/// Build the Variational Quantum Circuit IR and return the populated `Context`.
///
/// Architecture (4 qubits):
/// ```text
/// Layer 1 — Encoding:      RY(q0), RY(q1), RY(q2), RY(q3)
/// Layer 2 — Entanglement:  CX(q0,q1), CX(q2,q3)
/// Layer 3 — Parametrised:  RZ(q0), RZ(q1), RZ(q2), RZ(q3)
/// ```
pub fn build_vqc_context() -> Context {
    let mut ctx = Context::new();
    let qubit_ty = ctx.make_qubit_type();

    // ── Block with 4 qubit arguments ──
    let block = ctx.create_block();
    let q0 = ctx.create_block_arg(block, qubit_ty);
    let q1 = ctx.create_block_arg(block, qubit_ty);
    let q2 = ctx.create_block_arg(block, qubit_ty);
    let q3 = ctx.create_block_arg(block, qubit_ty);

    // ── Layer 1: RY angle encoding ──
    let (op0, ry0) = ctx.create_op(
        "quantum.ry", "quantum", vec![q0], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op0);

    let (op1, ry1) = ctx.create_op(
        "quantum.ry", "quantum", vec![q1], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op1);

    let (op2, ry2) = ctx.create_op(
        "quantum.ry", "quantum", vec![q2], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op2);

    let (op3, ry3) = ctx.create_op(
        "quantum.ry", "quantum", vec![q3], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op3);

    // ── Layer 2: CX entanglement ──
    let (op4, cx01) = ctx.create_op(
        "quantum.cx", "quantum", vec![ry0[0], ry1[0]], vec![qubit_ty, qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op4);

    let (op5, cx23) = ctx.create_op(
        "quantum.cx", "quantum", vec![ry2[0], ry3[0]], vec![qubit_ty, qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op5);

    // ── Layer 3: RZ parametrised rotations ──
    let (op6, _) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx01[0]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op6);

    let (op7, _) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx01[1]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op7);

    let (op8, _) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx23[0]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op8);

    let (op9, _) = ctx.create_op(
        "quantum.rz", "quantum", vec![cx23[1]], vec![qubit_ty],
        Attributes::new(), Location::unknown(),
    );
    ctx.add_op_to_block(block, op9);

    ctx
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Build both IRs and validate that the contexts are non-empty.
pub fn run(report: &mut TestReport) -> (Context, Context) {
    let cnn = build_cnn_context();
    report.check("Build CNN encoder IR (8 tensor ops)", cnn.ops.len() == 8);
    println!("    CNN: {} ops, {} values, {} blocks",
        cnn.ops.len(), cnn.values.len(), cnn.blocks.len());

    let vqc = build_vqc_context();
    report.check("Build VQC circuit IR (10 quantum ops)", vqc.ops.len() == 10);
    println!("    VQC: {} ops, {} values, {} blocks",
        vqc.ops.len(), vqc.values.len(), vqc.blocks.len());

    (cnn, vqc)
}
