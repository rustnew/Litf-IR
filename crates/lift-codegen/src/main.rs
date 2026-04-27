//! ============================================================================
//! lift-codegen — Define models & algorithms from Rust, emit .lif + .lith
//! ============================================================================
//!
//! Run: cargo run --bin lift-codegen
//!
//! This binary demonstrates LIFT's programmatic model construction.
//! When executed, it:
//!   1. Defines models using the ModelBuilder fluent API
//!   2. Generates .lif files (LIFT IR source)
//!   3. Generates .lith files (compilation config)
//!   4. Verifies the generated IR
//!   5. Analyses the model (FLOPs, memory)
//!   6. Runs optimization passes
//!   7. Exports to LLVM IR
//!
//! All files are written to the project root.
//! ============================================================================

use lift_core::model_builder::*;
use lift_core::types::DataType;
use lift_sim::analysis::analyze_module;
use lift_core::PassManager;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  LIFT Code Generator — Models from Rust                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. Phi-3-mini ──
    println!("── Generating Phi-3-mini ──");
    let phi3 = build_phi3();
    emit_and_validate("phi3_generated", phi3);

    // ── 2. MLP classifier ──
    println!("── Generating MLP Classifier ──");
    let mlp = build_mlp();
    emit_and_validate("mlp_generated", mlp);

    // ── 3. ResNet block ──
    println!("── Generating ResNet Block ──");
    let resnet = build_resnet_block();
    emit_and_validate("resnet_generated", resnet);

    // ── 4. Quantum VQE circuit ──
    println!("── Generating VQE Circuit ──");
    let vqe = build_vqe();
    emit_and_validate("vqe_generated", vqe);

    // ── 5. Generate config ──
    println!("── Generating Optimization Config ──");
    let config = build_lith_config(
        "llvm", "h100", "fp16",
        &[
            "canonicalize", "constant-folding", "dce",
            "tensor-fusion", "flash-attention", "cse",
            "quantisation-pass",
        ],
        Some(500_000_000_000),
        Some(80_000_000_000),
    );
    std::fs::write("examples/generated_optimize.lith", &config).unwrap();
    println!("  [WRITE] examples/generated_optimize.lith ({} bytes)", config.len());

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  All models generated, verified, analysed, and exported!    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: emit, verify, analyse, optimise, export
// ─────────────────────────────────────────────────────────────────────────────

fn emit_and_validate(name: &str, builder: ModelBuilder) {
    // 1. Write parseable .lif source
    let lif_text = builder.build_lif();
    let lif_path = format!("examples/{}.lif", name);
    std::fs::write(&lif_path, &lif_text).unwrap();
    println!("  [WRITE] {} ({} bytes)", lif_path, lif_text.len());

    // 2. Build context for verification/analysis/export
    let ctx = builder.build_context();

    // 3. Verify the context
    match lift_core::verifier::verify(&ctx) {
        Ok(()) => {
            println!("  [VERIFY] OK — {} ops, {} values", ctx.ops.len(), ctx.values.len());
        }
        Err(errors) => {
            println!("  [VERIFY] FAILED — {} errors:", errors.len());
            for e in &errors {
                println!("    - {}", e);
            }
        }
    }

    // 4. Analyse
    let report = analyze_module(&ctx);
    println!(
        "  [ANALYSE] FLOPs={}, Memory={}, Ops={}",
        format_flops(report.total_flops),
        format_bytes(report.total_memory_bytes),
        report.num_ops,
    );

    // 5. Check if quantum ops present
    let has_quantum = ctx.ops.iter().any(|(_, op)| ctx.strings.resolve(op.name).starts_with("quantum."));

    // 6. Optimise (default passes)
    let mut opt_ctx = ctx;
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
    pm.add_pass(Box::new(lift_opt::TensorFusion));
    let results = pm.run_all(&mut opt_ctx);
    let changed: Vec<_> = results.iter()
        .filter(|(_, r)| matches!(r, lift_core::PassResult::Changed))
        .map(|(n, _)| n.as_str())
        .collect();
    if changed.is_empty() {
        println!("  [OPTIMISE] No changes");
    } else {
        println!("  [OPTIMISE] Changed: {}", changed.join(", "));
    }

    // 7. Export LLVM
    let llvm_exporter = lift_export::LlvmExporter::new();
    let llvm_ir = llvm_exporter.export(&opt_ctx).unwrap();
    let llvm_path = format!("examples/{}.ll", name);
    std::fs::write(&llvm_path, &llvm_ir).unwrap();
    println!("  [EXPORT] {} ({} bytes)", llvm_path, llvm_ir.len());

    // 8. Export ONNX
    let onnx_exporter = lift_export::OnnxExporter::new();
    let onnx_ir = onnx_exporter.export(&opt_ctx).unwrap();
    let onnx_path = format!("examples/{}.onnx", name);
    std::fs::write(&onnx_path, &onnx_ir).unwrap();
    println!("  [EXPORT] {} ({} bytes)", onnx_path, onnx_ir.len());

    // 9. Export OpenQASM (if quantum)
    if has_quantum {
        let qasm_exporter = lift_export::QasmExporter::new();
        let qasm_ir = qasm_exporter.export(&opt_ctx).unwrap();
        let qasm_path = format!("examples/{}.qasm", name);
        std::fs::write(&qasm_path, &qasm_ir).unwrap();
        println!("  [EXPORT] {} ({} bytes)", qasm_path, qasm_ir.len());
    }

    println!();
}

// ─────────────────────────────────────────────────────────────────────────────
// Model definitions
// ─────────────────────────────────────────────────────────────────────────────

fn build_phi3() -> ModelBuilder {
    let f = DataType::FP32;
    ModelBuilder::new("phi3_mini")
        .function("embedding")
            .param("ids", tensor(&[1, 128], DataType::INT32))
            .param("table", tensor_2d(32064, 3072, f))
            .op("tensor.embedding", &["ids", "table"], "emb", tensor(&[1, 128, 3072], f))
            .returns("emb")
            .done()
        .function("layer")
            .param("x", tensor(&[1, 128, 3072], f))
            .param("ln1_w", tensor_1d(3072, f))
            .param("wq", tensor_2d(3072, 3072, f))
            .param("wk", tensor_2d(3072, 3072, f))
            .param("wv", tensor_2d(3072, 3072, f))
            .param("wo", tensor_2d(3072, 3072, f))
            .param("ln2_w", tensor_1d(3072, f))
            .param("w_gate", tensor_2d(3072, 8192, f))
            .param("w_up", tensor_2d(3072, 8192, f))
            .param("w_down", tensor_2d(8192, 3072, f))
            .op("tensor.rmsnorm", &["x", "ln1_w"], "n1", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wq"], "q", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wk"], "k", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wv"], "v", tensor(&[1, 128, 3072], f))
            .op("tensor.grouped_query_attention", &["q", "k", "v"], "attn", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["attn", "wo"], "ap", tensor(&[1, 128, 3072], f))
            .op("tensor.add", &["x", "ap"], "r1", tensor(&[1, 128, 3072], f))
            .op("tensor.rmsnorm", &["r1", "ln2_w"], "n2", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n2", "w_gate"], "gate", tensor(&[1, 128, 8192], f))
            .op("tensor.silu", &["gate"], "ga", tensor(&[1, 128, 8192], f))
            .op("tensor.matmul", &["n2", "w_up"], "up", tensor(&[1, 128, 8192], f))
            .op("tensor.mul", &["ga", "up"], "gated", tensor(&[1, 128, 8192], f))
            .op("tensor.matmul", &["gated", "w_down"], "dn", tensor(&[1, 128, 3072], f))
            .op("tensor.add", &["r1", "dn"], "r2", tensor(&[1, 128, 3072], f))
            .returns("r2")
            .done()
        .function("lm_head")
            .param("x", tensor(&[1, 128, 3072], f))
            .param("ln_w", tensor_1d(3072, f))
            .param("lm_w", tensor_2d(3072, 32064, f))
            .op("tensor.rmsnorm", &["x", "ln_w"], "normed", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["normed", "lm_w"], "logits", tensor(&[1, 128, 32064], f))
            .returns("logits")
            .done()
}

fn build_mlp() -> ModelBuilder {
    let f = DataType::FP32;
    ModelBuilder::new("mlp_classifier")
        .function("forward")
            .param("x", tensor(&[1, 784], f))
            .param("w1", tensor_2d(784, 512, f))
            .param("b1", tensor_1d(512, f))
            .param("w2", tensor_2d(512, 256, f))
            .param("b2", tensor_1d(256, f))
            .param("w3", tensor_2d(256, 10, f))
            .param("b3", tensor_1d(10, f))
            .op("tensor.matmul", &["x", "w1"], "h1", tensor(&[1, 512], f))
            .op("tensor.add", &["h1", "b1"], "h1b", tensor(&[1, 512], f))
            .op("tensor.relu", &["h1b"], "a1", tensor(&[1, 512], f))
            .op("tensor.matmul", &["a1", "w2"], "h2", tensor(&[1, 256], f))
            .op("tensor.add", &["h2", "b2"], "h2b", tensor(&[1, 256], f))
            .op("tensor.relu", &["h2b"], "a2", tensor(&[1, 256], f))
            .op("tensor.matmul", &["a2", "w3"], "h3", tensor(&[1, 10], f))
            .op("tensor.add", &["h3", "b3"], "h3b", tensor(&[1, 10], f))
            .op("tensor.softmax", &["h3b"], "out", tensor(&[1, 10], f))
            .returns("out")
            .done()
}

fn build_resnet_block() -> ModelBuilder {
    let f = DataType::FP32;
    ModelBuilder::new("resnet_block")
        .function("block")
            .param("x", tensor_4d(1, 64, 56, 56, f))
            .param("w1", tensor_4d(64, 64, 3, 3, f))
            .param("w2", tensor_4d(64, 64, 3, 3, f))
            .param("bn1_w", tensor_1d(64, f))
            .param("bn2_w", tensor_1d(64, f))
            .op("tensor.conv2d", &["x", "w1"], "c1", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.batchnorm", &["c1", "bn1_w"], "bn1", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.relu", &["bn1"], "a1", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.conv2d", &["a1", "w2"], "c2", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.batchnorm", &["c2", "bn2_w"], "bn2", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.add", &["x", "bn2"], "res", tensor_4d(1, 64, 56, 56, f))
            .op("tensor.relu", &["res"], "out", tensor_4d(1, 64, 56, 56, f))
            .returns("out")
            .done()
}

fn build_vqe() -> ModelBuilder {
    let q = ModelType::Qubit;
    ModelBuilder::new("vqe_circuit")
        .dialect("quantum")
        .function("ansatz")
            .param("q0", q.clone())
            .param("q1", q.clone())
            .op("quantum.ry", &["q0"], "q0a", q.clone())
            .op("quantum.ry", &["q1"], "q1a", q.clone())
            .op("quantum.cx", &["q0a", "q1a"], "q0b", q.clone())
            .op("quantum.rz", &["q0b"], "q0c", q.clone())
            .returns("q0c")
            .done()
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting utilities
// ─────────────────────────────────────────────────────────────────────────────

fn format_flops(f: u64) -> String {
    if f >= 1_000_000_000_000 { format!("{:.2} TFLOP", f as f64 / 1e12) }
    else if f >= 1_000_000_000 { format!("{:.2} GFLOP", f as f64 / 1e9) }
    else if f >= 1_000_000 { format!("{:.2} MFLOP", f as f64 / 1e6) }
    else { format!("{} FLOP", f) }
}

fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 { format!("{:.2} GiB", b as f64 / 1_073_741_824.0) }
    else if b >= 1_048_576 { format!("{:.2} MiB", b as f64 / 1_048_576.0) }
    else if b >= 1024 { format!("{:.2} KiB", b as f64 / 1024.0) }
    else { format!("{} B", b) }
}
