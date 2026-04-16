// ============================================================================
// step6_feedback.rs — Step 6: Feedback Loop (Budget, Energy, Topology)
// ============================================================================
//
// After prediction, LIFT can enforce resource budgets and estimate operational
// costs before any real execution. This step covers:
//
// 6.1 Noise modelling and circuit-level fidelity tracking:
//   Simulate the fidelity degradation gate-by-gate through the VQC to
//   identify which layer dominates the error (typically 2Q gates).
//
// 6.2 Device topology and routing cost:
//   Evaluate connectivity of different QPU architectures (grid, heavy-hex,
//   all-to-all, linear) and compute SWAP distances for qubit routing.
//
// 6.3 Energy and carbon estimation:
//   Estimate GPU power draw (TDP × PUE), cryogenic cooling for QPU, and
//   CO₂ emissions for both inference and training scenarios.
//
// 6.4 Budget enforcement (static + reactive):
//   Static budgets check hard limits (FLOPs, memory, fidelity).
//   Reactive budgets track consumption across VQE-style iterations and
//   halt compilation when a constraint is exhausted.
//
// If the user executes on real hardware, collected metrics can be compared
// to predictions for continuous cost-model improvement.
//
// ============================================================================

use lift_sim::AnalysisReport;
use lift_sim::QuantumAnalysis;

use crate::report::TestReport;

// ────────────────────────────────────────────────────────────────────────────
// 6.1 Noise modelling
// ────────────────────────────────────────────────────────────────────────────

/// Simulate gate-by-gate noise accumulation for the VQC circuit and verify
/// that fidelity degrades predictably and meets operational thresholds.
pub fn test_noise_modelling(report: &mut TestReport) {
    use lift_quantum::noise::{CircuitNoise, GateNoise, NoiseModel};

    // ── Individual noise models ──
    let depol = NoiseModel::Depolarizing { p: 0.001 };
    let fid_depol = depol.fidelity();
    println!("    Depolarizing(p=0.001): fidelity={:.6}", fid_depol);
    report.check("Depolarizing fidelity ~0.999", (fid_depol - 0.999).abs() < 0.001);

    let thermal = NoiseModel::ThermalRelaxation {
        t1_us: 100.0,
        t2_us: 80.0,
        gate_time_us: 0.3,
    };
    let fid_therm = thermal.fidelity();
    println!("    Thermal(T1=100, T2=80, t=0.3): fidelity={:.6}", fid_therm);
    report.check("Thermal fidelity in (0, 1]", fid_therm > 0.0 && fid_therm <= 1.0);

    // ── Composed noise ──
    let composed = depol.compose(&thermal);
    let fid_comp = composed.fidelity();
    println!("    Composed fidelity: {:.6}", fid_comp);
    report.check(
        "Composed fidelity <= min(individual)",
        fid_comp <= fid_depol && fid_comp <= fid_therm,
    );

    // ── Circuit-level noise for VQC (4×RY + 2×CX + 4×RZ) ──
    let mut circuit = CircuitNoise::new();
    let g1q = GateNoise::with_depolarizing(0.999, 0.02);
    let g2q = GateNoise::with_depolarizing(0.99, 0.3);

    // Layer 1: 4× RY (1Q gates)
    for _ in 0..4 {
        circuit.add_gate(&g1q, false);
    }
    let after_ry = circuit.total_fidelity;
    println!("    After 4× RY:  fidelity={:.6}", after_ry);

    // Layer 2: 2× CX (2Q gates — dominant error source)
    for _ in 0..2 {
        circuit.add_gate(&g2q, true);
    }
    let after_cx = circuit.total_fidelity;
    println!("    After 2× CX:  fidelity={:.6}", after_cx);

    // Layer 3: 4× RZ (1Q gates)
    for _ in 0..4 {
        circuit.add_gate(&g1q, false);
    }
    let final_fid = circuit.total_fidelity;
    println!("    After 4× RZ:  fidelity={:.6} (final)", final_fid);

    println!(
        "    Circuit totals: {} gates, {} 2Q gates",
        circuit.gate_count, circuit.two_qubit_count
    );

    report.check("Fidelity degrades: after_CX < after_RY", after_cx < after_ry);
    report.check("Total gates = 10", circuit.gate_count == 10);
    report.check("2Q gates = 2", circuit.two_qubit_count == 2);
    report.check("Meets 90% threshold", circuit.meets_threshold(0.90));
}

// ────────────────────────────────────────────────────────────────────────────
// 6.2 Device topology and routing
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate QPU topologies relevant to the hybrid model and verify
/// connectivity, shortest paths, and SWAP distances.
pub fn test_device_topology(report: &mut TestReport) {
    use lift_quantum::DeviceTopology;

    // ── Grid 2×2 (matches our 4-qubit VQC) ──
    let grid = DeviceTopology::grid(2, 2);
    println!("    Grid 2×2: {} qubits, {} edges", grid.num_qubits, grid.edges.len());
    report.check("Grid has 4 qubits", grid.num_qubits == 4);

    let conn_01 = grid.are_connected(0, 1);
    println!("    q0─q1 connected: {}", conn_01);
    report.check("Grid: q0─q1 connected", conn_01);

    if let Some(path) = grid.shortest_path(0, 3) {
        println!("    Path q0→q3: {:?} ({} SWAPs)", path, path.len().saturating_sub(2));
        report.check("Path q0→q3 found", true);
    } else {
        report.check("Path q0→q3 found", false);
    }

    let neighbors = grid.neighbors(0);
    println!("    Neighbours of q0: {:?}", neighbors);
    report.check("q0 has neighbours", !neighbors.is_empty());

    // ── Heavy-hex (IBM 127-qubit) ──
    let hh = DeviceTopology::heavy_hex(127);
    println!(
        "    Heavy-hex: {} qubits, {} edges, diameter {}",
        hh.num_qubits,
        hh.edges.len(),
        hh.diameter()
    );
    report.check("Heavy-hex has 127 qubits", hh.num_qubits == 127);

    // ── All-to-all (trapped-ion, full connectivity) ──
    let ata = DeviceTopology::all_to_all(4);
    println!(
        "    All-to-all(4): {} edges, avg connectivity {:.2}",
        ata.edges.len(),
        ata.avg_connectivity()
    );
    report.check("All-to-all(4) has 6 edges", ata.edges.len() == 6);

    let swap_ata = ata.swap_distance(0, 3);
    println!("    All-to-all SWAP q0→q3: {:?}", swap_ata);
    report.check("All-to-all: 0 SWAPs (direct)", swap_ata == Some(0));

    // ── Linear chain ──
    let linear = DeviceTopology::linear(4);
    let swap_lin = linear.swap_distance(0, 3);
    println!("    Linear(4) SWAP q0→q3: {:?}", swap_lin);
    report.check("Linear: q0→q3 needs 2 SWAPs", swap_lin == Some(2));
}

// ────────────────────────────────────────────────────────────────────────────
// 6.3 Energy and carbon estimation
// ────────────────────────────────────────────────────────────────────────────

/// Estimate energy consumption and CO₂ emissions for inference, training,
/// and quantum execution scenarios.
pub fn test_energy_estimation(cnn_report: &AnalysisReport, report: &mut TestReport) {
    let a100_cost = lift_sim::cost::CostModel::a100();
    let pred = lift_predict::predict_performance(cnn_report, &a100_cost);
    let energy = lift_sim::cost::EnergyModel::a100();

    // ── Single inference ──
    let joules = energy.energy_joules(pred.predicted_time_ms, 1);
    let kwh = energy.energy_kwh(pred.predicted_time_ms, 1);
    let co2 = energy.carbon_grams(pred.predicted_time_ms, 1);

    println!("    Single inference (A100):");
    println!("      Energy: {:.6} J ({:.10} kWh)", joules, kwh);
    println!("      CO₂:   {:.8} g", co2);
    report.check("Inference energy >= 0", joules >= 0.0);
    report.check("Inference CO₂ >= 0", co2 >= 0.0);

    // ── Training: 8 GPUs, 24 hours ──
    let train_ms = 24.0 * 3600.0 * 1000.0;
    let train_kwh = energy.energy_kwh(train_ms, 8);
    let train_co2_kg = energy.carbon_grams(train_ms, 8) / 1000.0;

    println!("    Training (8× A100, 24h):");
    println!("      Energy: {:.2} kWh", train_kwh);
    println!("      CO₂:   {:.2} kg", train_co2_kg);
    report.check("Training energy > 0", train_kwh > 0.0);

    // ── Quantum energy (cryogenics-dominated) ──
    let q_joules = energy.quantum_energy_joules(100.0, 4);
    println!("    Quantum (100 μs, 4 qubits): {:.6} J", q_joules);
    report.check("Quantum energy > 0", q_joules > 0.0);
}

// ────────────────────────────────────────────────────────────────────────────
// 6.4 Budget enforcement (static + reactive)
// ────────────────────────────────────────────────────────────────────────────

/// Test both static budget checks (pass/fail) and reactive budget tracking
/// across iterative VQE-style optimisation loops.
pub fn test_budget_enforcement(
    cnn_report: &AnalysisReport,
    vqc_analysis: &QuantumAnalysis,
    report: &mut TestReport,
) {
    use lift_sim::cost::{Budget, ReactiveBudget};

    // ── Static budget: generous (should pass) ──
    let generous = Budget {
        max_flops: Some(100_000_000_000),
        max_memory_bytes: Some(80_000_000_000),
        max_time_ms: Some(1000.0),
        min_fidelity: Some(0.50),
        max_circuit_depth: None,
    };
    report.check(
        "Generous FLOP budget OK",
        generous.check_flops(cnn_report.total_flops).is_ok(),
    );
    report.check(
        "Generous memory budget OK",
        generous.check_memory(cnn_report.total_memory_bytes).is_ok(),
    );
    report.check(
        "Generous fidelity OK",
        generous.check_fidelity(vqc_analysis.estimated_fidelity).is_ok(),
    );

    // ── Static budget: tight (should fail) ──
    let tight = Budget {
        max_flops: Some(100),
        max_memory_bytes: Some(100),
        max_time_ms: None,
        min_fidelity: Some(0.9999),
        max_circuit_depth: None,
    };
    report.check(
        "Tight FLOP budget FAILS",
        tight.check_flops(cnn_report.total_flops).is_err(),
    );
    report.check(
        "Tight memory budget FAILS",
        tight.check_memory(cnn_report.total_memory_bytes).is_err(),
    );

    // ── Reactive budget: VQE-style iteration loop ──
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: Some(1000.0),
        min_fidelity: Some(0.80),
        max_circuit_depth: None,
    };
    let mut tracker = ReactiveBudget::new(budget);

    let mut iters = 0u32;
    for i in 0..200 {
        tracker.consume(0, 0, 10.0, 0.999);
        if tracker.check_remaining().is_err() {
            println!("    Reactive budget exhausted at iteration {}", i);
            break;
        }
        iters = i + 1;
    }

    println!(
        "    Completed {} iterations, elapsed={:.0} ms, fidelity={:.6}",
        iters, tracker.elapsed_ms, tracker.current_fidelity
    );

    report.check("Reactive: stopped before 200 iters", iters < 200);
    report.check("Reactive: elapsed > 0", tracker.elapsed_ms > 0.0);

    let util = tracker.utilisation();
    if let Some(time_ratio) = util.time_ratio {
        println!("    Time utilisation: {:.1}%", time_ratio * 100.0);
        report.check("Time utilisation >= 90%", time_ratio >= 0.9);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test entry point
// ────────────────────────────────────────────────────────────────────────────

/// Run all Step 6 feedback tests: noise, topology, energy, budgets.
pub fn run(
    cnn_report: &AnalysisReport,
    vqc_analysis: &QuantumAnalysis,
    report: &mut TestReport,
) {
    test_noise_modelling(report);
    test_device_topology(report);
    test_energy_estimation(cnn_report, report);
    test_budget_enforcement(cnn_report, vqc_analysis, report);
}
