use lift_hybrid::ops::{HybridOp, AnsatzType, SyncPolicy, FeatureMap};
use lift_sim::cost::{EnergyModel, ReactiveBudget, Budget, CostModel, QuantumCostModel};

// ═══════════════════════════════════════════════════════════
// HybridOp tests — new ops
// ═══════════════════════════════════════════════════════════

#[test]
fn test_new_hybrid_ops_roundtrip() {
    let ops = [
        HybridOp::AdjointDifferentiation,
        HybridOp::StochasticParameterShift,
        HybridOp::VqcLayer,
        HybridOp::VqeAnsatz,
        HybridOp::QaoaLayer,
        HybridOp::QuantumKernel,
        HybridOp::GpuToQpu,
        HybridOp::QpuToGpu,
        HybridOp::MeasureExpectation,
        HybridOp::MeasureSamples,
    ];
    for op in &ops {
        let name = op.op_name();
        let recovered = HybridOp::from_name(name);
        assert_eq!(recovered.as_ref(), Some(op), "roundtrip failed for {:?}", op);
    }
}

#[test]
fn test_original_hybrid_ops_still_work() {
    let ops = [
        HybridOp::Encode, HybridOp::Decode,
        HybridOp::ParameterShift, HybridOp::FiniteDifference,
        HybridOp::SPSA, HybridOp::JointGradient,
        HybridOp::ClassicalPreprocess, HybridOp::QuantumPostprocess,
        HybridOp::HybridForward, HybridOp::HybridBackward,
        HybridOp::CoExecute,
    ];
    for op in &ops {
        assert_eq!(HybridOp::from_name(op.op_name()).as_ref(), Some(op));
    }
}

#[test]
fn test_is_gradient() {
    let grad_ops = [
        HybridOp::ParameterShift, HybridOp::FiniteDifference,
        HybridOp::SPSA, HybridOp::AdjointDifferentiation,
        HybridOp::StochasticParameterShift, HybridOp::JointGradient,
    ];
    for op in &grad_ops {
        assert!(op.is_gradient(), "{:?} should be gradient", op);
    }
    assert!(!HybridOp::Encode.is_gradient());
    assert!(!HybridOp::VqcLayer.is_gradient());
}

#[test]
fn test_is_variational() {
    let var_ops = [
        HybridOp::VqcLayer, HybridOp::VqeAnsatz,
        HybridOp::QaoaLayer, HybridOp::QuantumKernel,
    ];
    for op in &var_ops {
        assert!(op.is_variational(), "{:?} should be variational", op);
    }
    assert!(!HybridOp::ParameterShift.is_variational());
    assert!(!HybridOp::CoExecute.is_variational());
}

#[test]
fn test_from_name_unknown_hybrid() {
    assert!(HybridOp::from_name("hybrid.nonexistent").is_none());
    assert!(HybridOp::from_name("tensor.matmul").is_none());
}

// ═══════════════════════════════════════════════════════════
// AnsatzType / SyncPolicy / FeatureMap enum tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_ansatz_type_variants() {
    let types = [
        AnsatzType::HardwareEfficient,
        AnsatzType::StronglyEntangling,
        AnsatzType::TwoLocal,
        AnsatzType::UCCSD,
        AnsatzType::Custom,
    ];
    for t in &types {
        assert_eq!(t, t); // PartialEq works
    }
    assert_ne!(AnsatzType::HardwareEfficient, AnsatzType::UCCSD);
}

#[test]
fn test_sync_policy_variants() {
    assert_ne!(SyncPolicy::Blocking, SyncPolicy::Asynchronous);
    assert_ne!(SyncPolicy::Asynchronous, SyncPolicy::Pipeline);
}

#[test]
fn test_feature_map_variants() {
    let maps = [
        FeatureMap::ZZFeatureMap, FeatureMap::PauliFeatureMap,
        FeatureMap::AngleEncoding, FeatureMap::AmplitudeEncoding,
    ];
    for m in &maps {
        assert_eq!(m, m);
    }
}

// ═══════════════════════════════════════════════════════════
// EnergyModel tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_energy_a100_profile() {
    let model = EnergyModel::a100();
    assert_eq!(model.gpu_tdp_watts, 400.0);
    assert!(model.cooling_pue > 1.0);
}

#[test]
fn test_energy_h100_profile() {
    let model = EnergyModel::h100();
    assert!(model.gpu_tdp_watts > EnergyModel::a100().gpu_tdp_watts);
}

#[test]
fn test_energy_joules_positive() {
    let model = EnergyModel::a100();
    let j = model.energy_joules(1000.0, 1);
    assert!(j > 0.0);
}

#[test]
fn test_energy_joules_scales_with_gpus() {
    let model = EnergyModel::a100();
    let j1 = model.energy_joules(1000.0, 1);
    let j4 = model.energy_joules(1000.0, 4);
    assert!(j4 > j1);
}

#[test]
fn test_energy_joules_scales_with_time() {
    let model = EnergyModel::a100();
    let j1 = model.energy_joules(100.0, 1);
    let j2 = model.energy_joules(200.0, 1);
    assert!((j2 - 2.0 * j1).abs() < 1e-6);
}

#[test]
fn test_energy_kwh_conversion() {
    let model = EnergyModel::a100();
    let kwh = model.energy_kwh(3_600_000.0, 1); // 1 hour
    let joules = model.energy_joules(3_600_000.0, 1);
    assert!((kwh - joules / 3_600_000.0).abs() < 1e-6);
}

#[test]
fn test_carbon_grams() {
    let model = EnergyModel::a100();
    let co2 = model.carbon_grams(1000.0, 1);
    assert!(co2 > 0.0);
}

#[test]
fn test_quantum_energy() {
    let model = EnergyModel::a100();
    let j = model.quantum_energy_joules(100.0, 27);
    assert!(j > 0.0);
}

// ═══════════════════════════════════════════════════════════
// ReactiveBudget tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_reactive_budget_new() {
    let budget = Budget {
        max_flops: Some(1_000_000),
        max_memory_bytes: Some(1_000_000),
        max_time_ms: Some(100.0),
        min_fidelity: Some(0.9),
        max_circuit_depth: None,
    };
    let rb = ReactiveBudget::new(budget);
    assert_eq!(rb.consumed_flops, 0);
    assert_eq!(rb.elapsed_ms, 0.0);
    assert_eq!(rb.current_fidelity, 1.0);
}

#[test]
fn test_reactive_budget_consume() {
    let budget = Budget {
        max_flops: Some(1_000_000),
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(500_000, 1024, 10.0, 0.99);
    assert_eq!(rb.consumed_flops, 500_000);
    assert_eq!(rb.consumed_memory, 1024);
    assert!((rb.elapsed_ms - 10.0).abs() < 1e-10);
    assert!((rb.current_fidelity - 0.99).abs() < 1e-10);
}

#[test]
fn test_reactive_budget_check_ok() {
    let budget = Budget {
        max_flops: Some(1_000_000),
        max_memory_bytes: Some(1_000_000),
        max_time_ms: Some(100.0),
        min_fidelity: Some(0.9),
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(100_000, 500, 5.0, 0.99);
    assert!(rb.check_remaining().is_ok());
}

#[test]
fn test_reactive_budget_flop_exceeded() {
    let budget = Budget {
        max_flops: Some(100),
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(200, 0, 0.0, 1.0);
    assert!(rb.check_remaining().is_err());
}

#[test]
fn test_reactive_budget_memory_exceeded() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: Some(100),
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 200, 0.0, 1.0);
    assert!(rb.check_remaining().is_err());
}

#[test]
fn test_reactive_budget_time_exceeded() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: Some(10.0),
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 0, 20.0, 1.0);
    assert!(rb.check_remaining().is_err());
}

#[test]
fn test_reactive_budget_fidelity_exceeded() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: Some(0.9),
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 0, 0.0, 0.8);
    assert!(rb.check_remaining().is_err());
}

#[test]
fn test_reactive_budget_remaining_flops() {
    let budget = Budget {
        max_flops: Some(1000),
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(300, 0, 0.0, 1.0);
    assert_eq!(rb.remaining_flops(), Some(700));
}

#[test]
fn test_reactive_budget_remaining_time() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: Some(50.0),
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 0, 20.0, 1.0);
    let remaining = rb.remaining_time_ms().unwrap();
    assert!((remaining - 30.0).abs() < 1e-10);
}

#[test]
fn test_reactive_budget_unlimited() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(u64::MAX / 2, u64::MAX / 2, 1e12, 0.001);
    assert!(rb.check_remaining().is_ok());
    assert!(rb.remaining_flops().is_none());
    assert!(rb.remaining_time_ms().is_none());
}

#[test]
fn test_reactive_budget_utilisation() {
    let budget = Budget {
        max_flops: Some(1000),
        max_memory_bytes: Some(2000),
        max_time_ms: Some(100.0),
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(500, 1000, 50.0, 1.0);
    let util = rb.utilisation();
    assert!((util.flop_ratio.unwrap() - 0.5).abs() < 1e-10);
    assert!((util.memory_ratio.unwrap() - 0.5).abs() < 1e-10);
    assert!((util.time_ratio.unwrap() - 0.5).abs() < 1e-10);
}

#[test]
fn test_reactive_budget_peak_memory() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: Some(10000),
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 5000, 0.0, 1.0);
    rb.consume(0, 3000, 0.0, 1.0); // lower, should keep peak
    assert_eq!(rb.consumed_memory, 5000);
    rb.consume(0, 8000, 0.0, 1.0); // higher, update peak
    assert_eq!(rb.consumed_memory, 8000);
}

// ═══════════════════════════════════════════════════════════
// CostModel extended tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_h100_faster_than_a100() {
    let a100 = CostModel::a100();
    let h100 = CostModel::h100();
    let flops = 1_000_000_000_000_u64;
    assert!(h100.compute_time_ms(flops) < a100.compute_time_ms(flops));
}

#[test]
fn test_compute_bound_classification() {
    let model = CostModel::a100();
    // High FLOPs, low memory -> compute bound
    assert!(model.is_compute_bound(1_000_000_000, 100));
    // Low FLOPs, high memory -> memory bound
    assert!(!model.is_compute_bound(100, 1_000_000_000));
}

#[test]
fn test_num_gpus_needed() {
    let model = CostModel::a100();
    assert_eq!(model.num_gpus_needed(40_000_000_000), 1); // 40GB < 80GB
    assert_eq!(model.num_gpus_needed(160_000_000_000), 2); // 160GB > 80GB
}

// ═══════════════════════════════════════════════════════════
// QuantumCostModel extended tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_trapped_ion_model() {
    let model = QuantumCostModel::trapped_ion_default();
    assert!(model.fidelity_1q > model.fidelity_2q);
    assert!(model.gate_time_1q_us < model.gate_time_2q_us);
    assert!(model.t1_us > 100_000.0); // seconds-scale T1
}

#[test]
fn test_neutral_atom_model() {
    let model = QuantumCostModel::neutral_atom_default();
    assert_eq!(model.num_qubits, 256);
    assert!(model.gate_time_1q_us < 1.0);
}

#[test]
fn test_trapped_ion_higher_fidelity_than_superconducting() {
    let sc = QuantumCostModel::superconducting_default();
    let ti = QuantumCostModel::trapped_ion_default();
    assert!(ti.fidelity_2q > sc.fidelity_2q);
}

#[test]
fn test_decoherence_decreases_with_time() {
    let model = QuantumCostModel::superconducting_default();
    let f1 = model.decoherence_fidelity(1.0);
    let f2 = model.decoherence_fidelity(10.0);
    assert!(f1 > f2);
}
