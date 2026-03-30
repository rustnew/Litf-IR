use lift_core::types::*;
use lift_core::context::Context;
use lift_core::printer::print_ir;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;
use lift_quantum::gates::{QuantumGate, Provider};
use lift_quantum::topology::DeviceTopology;
use lift_quantum::kraus::{ComplexMatrix, KrausChannel};
use lift_quantum::qec::{QecCode, QecAnalysis};
use lift_quantum::noise::{NoiseModel, GateNoise, CircuitNoise};
use lift_sim::cost::{CostModel, QuantumCostModel, Budget, EnergyModel, ReactiveBudget};
use lift_hybrid::ops::HybridOp;
use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// ═══════════════════════════════════════════════════════════
// Shape inference — more ops
// ═══════════════════════════════════════════════════════════

#[test]
fn test_shape_embedding() {
    let indices = mk(vec![4, 10], DataType::INT32);
    let table = mk(vec![1000, 256], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Embedding, &[&indices, &table]);
    assert!(result.is_ok());
}

#[test]
fn test_shape_softmax() {
    let a = mk(vec![2, 10], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Softmax, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_layer_norm() {
    let a = mk(vec![2, 8, 64], DataType::FP32);
    let scale = mk(vec![64], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::LayerNorm, &[&a, &scale]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_rms_norm() {
    let a = mk(vec![4, 128], DataType::FP16);
    let scale = mk(vec![128], DataType::FP16);
    let result = ShapeInference::infer_output_shape(&TensorOp::RMSNorm, &[&a, &scale]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_div() {
    let a = mk(vec![3, 4], DataType::FP32);
    let b = mk(vec![3, 4], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Div, &[&a, &b]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_sub() {
    let a = mk(vec![5], DataType::FP32);
    let b = mk(vec![5], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Sub, &[&a, &b]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_neg() {
    let a = mk(vec![2, 3], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Neg, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_sigmoid() {
    let a = mk(vec![1, 100], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Sigmoid, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_tanh() {
    let a = mk(vec![8, 64], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Tanh, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_silu() {
    let a = mk(vec![4, 256], DataType::FP16);
    let result = ShapeInference::infer_output_shape(&TensorOp::SiLU, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

// ═══════════════════════════════════════════════════════════
// FLOPs — more ops
// ═══════════════════════════════════════════════════════════

#[test]
fn test_flops_linear() {
    let x = mk(vec![1, 128], DataType::FP32);
    let w = mk(vec![128, 64], DataType::FP32);
    let b = mk(vec![64], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Linear, &[&x, &w, &b]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 0);
}

#[test]
fn test_flops_softmax() {
    let a = mk(vec![2, 100], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Softmax, &[&a]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 0);
}

#[test]
fn test_flops_layer_norm() {
    let a = mk(vec![2, 64], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::LayerNorm, &[&a]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 0);
}

#[test]
fn test_flops_add() {
    let a = mk(vec![10, 10], DataType::FP32);
    let b = mk(vec![10, 10], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Add, &[&a, &b]);
    assert_eq!(flops, Some(100));
}

#[test]
fn test_flops_gelu() {
    let a = mk(vec![4, 4], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::GeLU, &[&a]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 16); // more than N due to approximation
}

// ═══════════════════════════════════════════════════════════
// Memory bytes
// ═══════════════════════════════════════════════════════════

#[test]
fn test_memory_linear() {
    let x = mk(vec![1, 128], DataType::FP32);
    let w = mk(vec![128, 64], DataType::FP32);
    let b = mk(vec![64], DataType::FP32);
    let mem = ShapeInference::compute_memory_bytes(&TensorOp::Linear, &[&x, &w, &b]);
    assert!(mem.is_some());
    assert!(mem.unwrap() > 0);
}

#[test]
fn test_memory_bf16() {
    let a = mk(vec![100], DataType::BF16);
    let mem = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&a]);
    assert_eq!(mem, Some(200)); // 100 * 2 bytes
}

// ═══════════════════════════════════════════════════════════
// Noise model — additional coverage
// ═══════════════════════════════════════════════════════════

#[test]
fn test_noise_ideal_fidelity() {
    assert_eq!(NoiseModel::Ideal.fidelity(), 1.0);
}

#[test]
fn test_noise_depolarizing_ranges() {
    for &p in &[0.0, 0.01, 0.05, 0.1, 0.5, 1.0] {
        let model = NoiseModel::Depolarizing { p };
        let f = model.fidelity();
        assert!(f >= 0.0 && f <= 1.0, "p={}, f={}", p, f);
    }
}

#[test]
fn test_noise_bit_flip_ranges() {
    for &p in &[0.0, 0.01, 0.1, 0.5] {
        let model = NoiseModel::BitFlip { p };
        let f = model.fidelity();
        assert!(f >= 0.0 && f <= 1.0);
    }
}

#[test]
fn test_gate_noise_with_depolarizing() {
    let gn = GateNoise::with_depolarizing(0.999, 0.02);
    assert_eq!(gn.gate_fidelity, 0.999);
    assert_eq!(gn.gate_time_us, 0.02);
}

#[test]
fn test_circuit_noise_empty() {
    let cn = CircuitNoise::new();
    assert_eq!(cn.gate_count, 0);
    assert_eq!(cn.two_qubit_count, 0);
    assert_eq!(cn.circuit_depth, 0);
    assert_eq!(cn.total_fidelity, 1.0);
    assert_eq!(cn.total_time_us, 0.0);
}

// ═══════════════════════════════════════════════════════════
// Topology — shortest path
// ═══════════════════════════════════════════════════════════

#[test]
fn test_shortest_path_linear() {
    let topo = DeviceTopology::linear(5);
    let path = topo.shortest_path(0, 4);
    assert!(path.is_some());
    let p = path.unwrap();
    assert_eq!(p.len(), 5); // 0-1-2-3-4
    assert_eq!(p[0], 0);
    assert_eq!(p[4], 4);
}

#[test]
fn test_shortest_path_self() {
    let topo = DeviceTopology::linear(3);
    let path = topo.shortest_path(1, 1);
    assert!(path.is_some());
    assert_eq!(path.unwrap().len(), 1);
}

#[test]
fn test_shortest_path_all_to_all() {
    let topo = DeviceTopology::all_to_all(5);
    let path = topo.shortest_path(0, 4);
    assert!(path.is_some());
    assert_eq!(path.unwrap().len(), 2); // direct connection
}

// ═══════════════════════════════════════════════════════════
// Complex matrix — more operations
// ═══════════════════════════════════════════════════════════

#[test]
fn test_matrix_scale() {
    let mut m = ComplexMatrix::identity(2);
    m.set(0, 0, (2.0, 0.0));
    m.set(1, 1, (3.0, 0.0));
    let tr = m.trace().unwrap();
    assert_eq!(tr, (5.0, 0.0));
}

#[test]
fn test_matrix_multiply_square() {
    let mut a = ComplexMatrix::new(2, 2);
    a.set(0, 0, (1.0, 0.0));
    a.set(0, 1, (2.0, 0.0));
    a.set(1, 0, (3.0, 0.0));
    a.set(1, 1, (4.0, 0.0));
    let id = ComplexMatrix::identity(2);
    let result = a.mul(&id).unwrap();
    assert_eq!(result.get(0, 0), (1.0, 0.0));
    assert_eq!(result.get(1, 1), (4.0, 0.0));
}

// ═══════════════════════════════════════════════════════════
// Kraus channel — fidelity bounds
// ═══════════════════════════════════════════════════════════

#[test]
fn test_kraus_fidelity_bounds() {
    for &p in &[0.0, 0.01, 0.05, 0.1, 0.3, 0.5] {
        let ch = KrausChannel::depolarizing(p, 1);
        let f = ch.average_gate_fidelity();
        assert!(f >= 0.0 && f <= 1.0, "p={}, f={}", p, f);
    }
}

#[test]
fn test_kraus_amplitude_damping_bounds() {
    for &g in &[0.0, 0.01, 0.1, 0.5, 1.0] {
        let ch = KrausChannel::amplitude_damping(g);
        let f = ch.average_gate_fidelity();
        assert!(f >= 0.0 && f <= 1.0, "gamma={}, f={}", g, f);
    }
}

// ═══════════════════════════════════════════════════════════
// QEC — various codes
// ═══════════════════════════════════════════════════════════

#[test]
fn test_surface_code_distances() {
    for d in [3, 5, 7, 9, 11] {
        let code = QecCode::SurfaceCode { distance: d };
        assert_eq!(code.code_distance(), d);
        assert_eq!(code.physical_per_logical(), d * d);
    }
}

#[test]
fn test_repetition_code_distances() {
    for d in [3, 5, 7, 11] {
        let code = QecCode::RepetitionCode { distance: d };
        assert_eq!(code.physical_per_logical(), d);
    }
}

// ═══════════════════════════════════════════════════════════
// CostModel — comparative
// ═══════════════════════════════════════════════════════════

#[test]
fn test_h100_higher_bandwidth_than_a100() {
    assert!(CostModel::h100().memory_bandwidth_gb_s > CostModel::a100().memory_bandwidth_gb_s);
}

#[test]
fn test_h100_more_flops_than_a100() {
    assert!(CostModel::h100().flops_per_second > CostModel::a100().flops_per_second);
}

#[test]
fn test_energy_h100_higher_tdp() {
    assert!(EnergyModel::h100().gpu_tdp_watts > EnergyModel::a100().gpu_tdp_watts);
}

// ═══════════════════════════════════════════════════════════
// ReactiveBudget — multiple consume calls
// ═══════════════════════════════════════════════════════════

#[test]
fn test_reactive_budget_flop_accumulation() {
    let budget = Budget {
        max_flops: Some(1000),
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    for _ in 0..10 {
        rb.consume(100, 0, 0.0, 1.0);
    }
    assert_eq!(rb.consumed_flops, 1000);
    assert!(rb.check_remaining().is_ok());
    rb.consume(1, 0, 0.0, 1.0);
    assert!(rb.check_remaining().is_err());
}

// ═══════════════════════════════════════════════════════════
// Encoding — edge cases
// ═══════════════════════════════════════════════════════════

#[test]
fn test_encoding_amplitude_power_of_2() {
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(16), 4);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(64), 6);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(1024), 10);
}

#[test]
fn test_encoding_config_clone() {
    let c = EncodingConfig::new(EncodingStrategy::IQPEncoding, 8);
    let c2 = c.clone();
    assert_eq!(c, c2);
}

// ═══════════════════════════════════════════════════════════
// Provider enum coverage
// ═══════════════════════════════════════════════════════════

#[test]
fn test_all_providers_have_basis() {
    let providers = [
        Provider::IbmEagle, Provider::IbmKyoto,
        Provider::Rigetti, Provider::IonQ,
        Provider::Quantinuum, Provider::Simulator,
    ];
    for p in &providers {
        let basis = QuantumGate::native_basis(*p);
        assert!(!basis.is_empty(), "{:?} has empty basis", p);
    }
}

#[test]
fn test_provider_debug() {
    let p = Provider::IbmEagle;
    let s = format!("{:?}", p);
    assert!(s.contains("IbmEagle"));
}

// ═══════════════════════════════════════════════════════════
// Context basic operations
// ═══════════════════════════════════════════════════════════

#[test]
fn test_context_new_empty() {
    let ctx = Context::new();
    assert!(ctx.modules.is_empty());
    assert!(ctx.ops.is_empty());
    assert!(ctx.values.is_empty());
    assert!(ctx.blocks.is_empty());
}

#[test]
fn test_context_intern_string() {
    let mut ctx = Context::new();
    let id1 = ctx.strings.intern("hello");
    let id2 = ctx.strings.intern("hello");
    assert_eq!(id1, id2);
    assert_eq!(ctx.strings.resolve(id1), "hello");
}

#[test]
fn test_context_intern_different_strings() {
    let mut ctx = Context::new();
    let id1 = ctx.strings.intern("a");
    let id2 = ctx.strings.intern("b");
    assert_ne!(id1, id2);
}

#[test]
fn test_print_ir_returns_string() {
    let ctx = Context::new();
    let s = print_ir(&ctx);
    assert!(s.len() < 1000); // should be small for empty context
}
