// ============================================================================
// step5_export.rs — Step 5: Export to Backend Code
// ============================================================================
//
// LIFT generates executable code from the optimised IR:
//
//   - **CNN → LLVM IR**: kernels for GPU/CPU execution (cuBLAS for matmuls,
//     cuDNN for convolutions, fused kernels where possible).
//   - **VQC → OpenQASM 3.0**: quantum circuit with native gates for the
//     target QPU (after decomposition, mapping, and SWAP insertion).
//
// Equivalent CLI:
//   lift export opt.lif --backend llvm  -o cnn_kernel.ll
//   lift export opt.lif --backend qasm  -o vqc_circuit.qasm
//
// ============================================================================

use lift_core::Context;

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// LLVM IR export (classical / tensor dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Export the CNN context to LLVM IR and validate the output.
pub fn export_llvm(ctx: &Context, report: &mut TestReport) {
    let exporter = lift_export::LlvmExporter::new();
    let result = exporter.export(ctx);

    match &result {
        Ok(ir) => println!("    LLVM IR: {} bytes", ir.len()),
        Err(e) => println!("    LLVM export error: {}", e),
    }

    report.check("LLVM export succeeds", result.is_ok());
    if let Ok(ir) = &result {
        report.check("LLVM IR non-empty", !ir.is_empty());
    }
}

// ────────────────────────────────────────────────────────────────────────────
// OpenQASM 3.0 export (quantum dialect)
// ────────────────────────────────────────────────────────────────────────────

/// Export the VQC context to OpenQASM 3.0 and validate the output.
pub fn export_qasm(ctx: &Context, report: &mut TestReport) {
    let exporter = lift_export::QasmExporter::new();
    let result = exporter.export(ctx);

    match &result {
        Ok(qasm) => println!("    OpenQASM 3.0: {} bytes", qasm.len()),
        Err(e) => println!("    QASM export error: {}", e),
    }

    report.check("QASM export succeeds", result.is_ok());
    if let Ok(qasm) = &result {
        report.check("QASM output non-empty", !qasm.is_empty());
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 5 export tests.
pub fn run(cnn_ctx: &Context, vqc_ctx: &Context, report: &mut TestReport) {
    export_llvm(cnn_ctx, report);
    export_qasm(vqc_ctx, report);
}
