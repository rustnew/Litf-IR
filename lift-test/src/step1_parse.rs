// ============================================================================
// step1_parse.rs — Step 1: Parse & Verify
// ============================================================================
//
// LIFT reads the `.lif` file and builds an SSA (Static Single Assignment)
// graph. This step validates:
//
//   - **Lexing / Parsing**: transforms source text into an AST.
//   - **SSA construction**: each value (%img, %q0, …) is defined exactly once.
//   - **Type verification**: operation inputs/outputs have compatible types
//     (e.g. `tensor.conv2d` receives tensors of compatible dimensions).
//   - **Qubit linearity**: no qubit is consumed more than once (no-cloning).
//
// If everything is valid, LIFT holds a complete IR representing the CNN,
// the quantum circuit, and their hybrid linkage.
//
// Equivalent CLI:
//   lift verify pneumonia.lif
//
// ============================================================================

use lift_core::Context;

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// .lif file parsing via lift-ast
// ────────────────────────────────────────────────────────────────────────────

/// Parse a `.lif` source file into a LIFT `Context`.
///
/// Runs the full pipeline: Lexer → Parser → IrBuilder.
/// Returns `Some(Context)` on success, `None` on any error (with diagnostics
/// printed to stdout).
pub fn parse_lif_file(path: &str) -> Option<Context> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("    Could not read {}: {}", path, e);
            return None;
        }
    };

    // Lexing
    let mut lexer = lift_ast::Lexer::new(&source);
    let tokens = lexer.tokenize().to_vec();
    if !lexer.errors().is_empty() {
        println!("    Lexer errors in {}: {:?}", path, lexer.errors());
        return None;
    }
    println!("    Lexed {} tokens from {}", tokens.len(), path);

    // Parsing
    let mut parser = lift_ast::Parser::new(tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => {
            println!("    Parse errors in {}: {:?}", path, e);
            return None;
        }
    };

    // IR construction
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

// ────────────────────────────────────────────────────────────────────────────
// IR verification
// ────────────────────────────────────────────────────────────────────────────

/// Verify an IR context for SSA correctness, type consistency, and qubit
/// linearity. Prints detailed diagnostics on failure.
pub fn verify_ir(ctx: &Context, label: &str) -> bool {
    match lift_core::verifier::verify(ctx) {
        Ok(()) => {
            println!("    {} IR: PASSED (SSA + types + linearity OK)", label);
            true
        }
        Err(errs) => {
            println!("    {} IR: {} error(s):", label, errs.len());
            for e in &errs {
                println!("      - {}", e);
            }
            false
        }
    }
}

/// Print human-readable IR and return the output string.
pub fn print_ir(ctx: &Context, label: &str) -> String {
    let output = lift_core::printer::print_ir(ctx);
    println!("    {} IR: {} chars", label, output.len());
    output
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 1 tests: parse `.lif` files, verify IR, print IR.
pub fn run(
    cnn_ctx: &Context,
    vqc_ctx: &Context,
    report: &mut TestReport,
) {
    // 1a. Parse .lif files from disk
    let cnn_parsed = parse_lif_file("examples/cnn_encoder.lif");
    report.check("Parse CNN encoder .lif", cnn_parsed.is_some());

    let vqc_parsed = parse_lif_file("examples/quantum_vqc.lif");
    report.check("Parse quantum VQC .lif", vqc_parsed.is_some());

    // 1b. Verify programmatically-built IR
    let cnn_ok = verify_ir(cnn_ctx, "CNN");
    report.check("Verify CNN IR (SSA + types)", cnn_ok);

    let vqc_ok = verify_ir(vqc_ctx, "VQC");
    report.check("Verify VQC IR (SSA + linearity)", vqc_ok);

    // 1c. Print IR
    let cnn_ir = print_ir(cnn_ctx, "CNN");
    report.check("Print CNN IR (non-empty)", !cnn_ir.is_empty());

    let vqc_ir = print_ir(vqc_ctx, "VQC");
    report.check("Print VQC IR (non-empty)", !vqc_ir.is_empty());
}
