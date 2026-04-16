// ============================================================================
// step4_predict.rs — Step 4: Performance Prediction (Roofline + Quantum)
// ============================================================================
//
// LIFT uses two complementary models to predict execution characteristics
// before any code runs on real hardware:
//
// 4.1 Classical roofline model:
//   From the FLOPs and total memory of the CNN, compute the theoretical
//   execution time on the target GPU (A100 or H100). Determine whether the
//   workload is compute-bound or memory-bound.
//
// 4.2 Quantum prediction model:
//   From the circuit depth, 1Q/2Q gate counts, and QPU noise parameters,
//   estimate the final fidelity and the number of shots required to reach
//   a given statistical precision.
//
// If the user defined budgets (max_latency_ms, min_fidelity), LIFT compares
// predictions to constraints and warns (or refuses to compile) if they are
// not satisfied.
//
// Equivalent CLI:
//   lift predict opt.lif --device a100 --quantum-device ibm_kyoto
//
// ============================================================================

use lift_sim::AnalysisReport;
use lift_sim::QuantumAnalysis;

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// Classical GPU roofline prediction
// ────────────────────────────────────────────────────────────────────────────

/// Predict CNN performance on NVIDIA A100 and H100 using the roofline model.
///
/// Reports: compute time, memory time, predicted (max) time, arithmetic
/// intensity, bottleneck type, memory fit, and GPU count.
pub fn predict_gpu(report_data: &AnalysisReport, report: &mut TestReport) {
    let a100 = lift_sim::cost::CostModel::a100();
    let h100 = lift_sim::cost::CostModel::h100();

    let pred_a100 = lift_predict::predict_performance(report_data, &a100);
    let pred_h100 = lift_predict::predict_performance(report_data, &h100);

    println!("    A100 prediction:");
    println!("      Compute time:  {:.6} ms", pred_a100.compute_time_ms);
    println!("      Memory time:   {:.6} ms", pred_a100.memory_time_ms);
    println!("      Predicted:     {:.6} ms", pred_a100.predicted_time_ms);
    println!("      Arith intens:  {:.2} FLOP/byte", pred_a100.arithmetic_intensity);
    println!("      Bottleneck:    {}", pred_a100.bottleneck);

    println!("    H100 prediction:");
    println!("      Predicted:     {:.6} ms", pred_h100.predicted_time_ms);
    if pred_h100.predicted_time_ms > 0.0 {
        println!(
            "      Speedup vs A100: {:.2}x",
            pred_a100.predicted_time_ms / pred_h100.predicted_time_ms
        );
    }

    report.check("A100 predicted time >= 0", pred_a100.predicted_time_ms >= 0.0);
    report.check("H100 predicted time >= 0", pred_h100.predicted_time_ms >= 0.0);
    report.check(
        "Bottleneck is 'compute' or 'memory'",
        pred_a100.bottleneck == "compute" || pred_a100.bottleneck == "memory",
    );

    // Memory fit check
    let fits = a100.fits_in_memory(report_data.total_memory_bytes);
    let gpus = a100.num_gpus_needed(report_data.total_memory_bytes);
    println!("    Fits in 1 A100: {} ({} GPU(s) needed)", fits, gpus);
    report.check("CNN fits in 1 GPU", fits);
}

// ────────────────────────────────────────────────────────────────────────────
// Quantum fidelity and shot prediction
// ────────────────────────────────────────────────────────────────────────────

/// Predict VQC performance on superconducting (IBM-like) and trapped-ion
/// (IonQ-like) quantum processors.
///
/// Reports: estimated fidelity, circuit execution time, required shot count
/// for 1% precision, and total execution time.
pub fn predict_quantum(analysis: &QuantumAnalysis, report: &mut TestReport) {
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

    report.check(
        "SC fidelity in (0, 1]",
        pred_sc.estimated_fidelity > 0.0 && pred_sc.estimated_fidelity <= 1.0,
    );
    report.check(
        "TI fidelity in (0, 1]",
        pred_ti.estimated_fidelity > 0.0 && pred_ti.estimated_fidelity <= 1.0,
    );
    report.check(
        "TI fidelity > SC fidelity (higher gate fidelity)",
        pred_ti.estimated_fidelity > pred_sc.estimated_fidelity,
    );
    report.check("Shot count > 0", pred_sc.num_shots_for_precision > 0);
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 4 prediction tests.
pub fn run(
    cnn_report: &AnalysisReport,
    vqc_analysis: &QuantumAnalysis,
    report: &mut TestReport,
) {
    predict_gpu(cnn_report, report);
    predict_quantum(vqc_analysis, report);
}
