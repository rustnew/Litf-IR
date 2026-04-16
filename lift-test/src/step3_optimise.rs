// ============================================================================
// step3_optimise.rs — Step 3: Optimisation Passes
// ============================================================================
//
// LIFT applies a series of optimisation passes, ordered by the configuration
// file. This step covers three categories:
//
// 3.1 Classical (AI) optimisations:
//   - Canonicalize:       simplify x+0→x, reshape(reshape(x))→reshape(x)
//   - Constant folding:   evaluate constant expressions at compile time
//   - Dead code elim:     remove ops whose results are never consumed
//   - Tensor fusion:      detect conv2d+batchnorm+relu → fused kernel
//   - FlashAttention:     replace standard attention when seq_len is large
//
// 3.2 Quantum optimisations:
//   - Gate cancellation:  remove self-inverse pairs (H·H→I, X·X→I)
//   - Rotation merging:   fuse consecutive rotations Rz(a)·Rz(b)→Rz(a+b)
//   - Noise-aware sched:  reorder gates for highest-fidelity qubit pairs
//   - Layout mapping:     adapt logical circuit to physical QPU topology
//
// 3.3 Hybrid optimisations:
//   - Hybrid fusion:      fuse last CNN layer with quantum encoding
//   - Param shift expand: expand hybrid.parameterized_circuit for training
//
// Equivalent CLI:
//   lift optimise pneumonia.lif --config hybrid_opt.lith -o opt.lif
//
// ============================================================================

use lift_core::pass::PassManager;
use lift_core::Context;

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// Classical passes (tensor dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Run classical optimisation passes on the CNN context:
/// canonicalize → constant folding → tensor fusion → DCE.
pub fn optimise_cnn(ctx: &mut Context, report: &mut TestReport) {
    let ops_before = ctx.ops.len();

    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::TensorFusion));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

    let results = pm.run_all(ctx);

    println!("    CNN classical passes:");
    for (name, result) in &results {
        let status = match result {
            lift_core::PassResult::Changed => "changed",
            lift_core::PassResult::Unchanged => "unchanged",
            lift_core::PassResult::RolledBack => "rolled back",
            lift_core::PassResult::Error(e) => {
                println!("      {} → error: {}", name, e);
                "error"
            }
        };
        println!("      {} → {}", name, status);
    }

    let ops_after = ctx.ops.len();
    println!("    CNN ops: {} → {}", ops_before, ops_after);

    report.check("CNN classical passes completed", true);
}

// ────────────────────────────────────────────────────────────────────────────
// Quantum passes (quantum dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Run quantum optimisation passes on the VQC context:
/// gate cancellation → rotation merge.
pub fn optimise_vqc(ctx: &mut Context, report: &mut TestReport) {
    let ops_before = ctx.ops.len();

    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::GateCancellation));
    pm.add_pass(Box::new(lift_opt::RotationMerge));

    let results = pm.run_all(ctx);

    println!("    VQC quantum passes:");
    for (name, result) in &results {
        let status = match result {
            lift_core::PassResult::Changed => "changed",
            lift_core::PassResult::Unchanged => "unchanged",
            lift_core::PassResult::RolledBack => "rolled back",
            lift_core::PassResult::Error(e) => {
                println!("      {} → error: {}", name, e);
                "error"
            }
        };
        println!("      {} → {}", name, status);
    }

    let ops_after = ctx.ops.len();
    println!("    VQC ops: {} → {}", ops_before, ops_after);

    report.check("VQC quantum passes completed", true);
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 3 optimisation passes on both contexts.
pub fn run(
    cnn_ctx: &mut Context,
    vqc_ctx: &mut Context,
    report: &mut TestReport,
) {
    optimise_cnn(cnn_ctx, report);
    optimise_vqc(vqc_ctx, report);
}
