use lift_core::types::*;
use lift_core::context::Context;
use lift_core::attributes::{Attributes, Attribute};
use lift_core::printer::print_ir;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;
use lift_quantum::gates::QuantumGate;
use lift_quantum::topology::DeviceTopology;
use lift_quantum::kraus::{ComplexMatrix, KrausChannel};
use lift_quantum::qec::{QecCode, QecAnalysis};
use lift_sim::cost::{CostModel, QuantumCostModel, Budget, EnergyModel, ReactiveBudget};

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// ═══════════════════════════════════════════════════════════
// Edge cases: shape inference with empty / scalar inputs
// ═══════════════════════════════════════════════════════════

#[test]
fn test_shape_infer_no_inputs() {
    let result = ShapeInference::infer_output_shape(&TensorOp::Add, &[]);
    assert!(result.is_err());
}

#[test]
fn test_shape_infer_wrong_input_count() {
    let a = mk(vec![2, 3], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Add, &[&a]);
    assert!(result.is_err());
}

#[test]
fn test_shape_matmul_incompatible() {
    let a = mk(vec![2, 3], DataType::FP32);
    let b = mk(vec![5, 4], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]);
    assert!(result.is_err());
}

#[test]
fn test_shape_scalar_relu() {
    let a = mk(vec![1], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::ReLU, &[&a]).unwrap();
    assert_eq!(result[0].shape.len(), 1);
    assert_eq!(result[0].shape[0].static_value(), Some(1));
}

#[test]
fn test_shape_broadcast_1d() {
    let a = mk(vec![1], DataType::FP32);
    let b = mk(vec![5], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Add, &[&a, &b]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(5));
}

#[test]
fn test_shape_broadcast_multidim() {
    let a = mk(vec![2, 1, 4], DataType::FP32);
    let b = mk(vec![1, 3, 4], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Mul, &[&a, &b]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(2));
    assert_eq!(result[0].shape[1].static_value(), Some(3));
    assert_eq!(result[0].shape[2].static_value(), Some(4));
}

// ═══════════════════════════════════════════════════════════
// Edge cases: quantum gates
// ═══════════════════════════════════════════════════════════

#[test]
fn test_gate_from_name_format() {
    assert!(QuantumGate::from_name("quantum.h").is_some());
    assert!(QuantumGate::from_name("H").is_none());
    assert!(QuantumGate::from_name("h").is_none());
}

#[test]
fn test_gate_from_name_empty() {
    assert!(QuantumGate::from_name("").is_none());
}

#[test]
fn test_gate_from_name_garbage() {
    assert!(QuantumGate::from_name("not_a_gate").is_none());
    assert!(QuantumGate::from_name("tensor.matmul").is_none());
}

#[test]
fn test_all_gates_have_names() {
    let gates = [
        QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::S, QuantumGate::Sdg, QuantumGate::T, QuantumGate::Tdg,
        QuantumGate::SX, QuantumGate::RX, QuantumGate::RY, QuantumGate::RZ,
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::CY, QuantumGate::SWAP,
        QuantumGate::ISWAP, QuantumGate::ECR, QuantumGate::ZZ,
        QuantumGate::XX, QuantumGate::YY, QuantumGate::CCX, QuantumGate::CSWAP,
        QuantumGate::Measure, QuantumGate::MeasureAll, QuantumGate::Reset,
        QuantumGate::Barrier,
    ];
    for g in &gates {
        let name = g.op_name();
        assert!(!name.is_empty());
        let recovered = QuantumGate::from_name(name);
        assert_eq!(recovered.as_ref(), Some(g));
    }
}

#[test]
fn test_self_inverse_known() {
    // Only test gates we're confident are self-inverse
    let self_inv = [
        QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::SWAP,
        QuantumGate::Rx180,
    ];
    for g in &self_inv {
        assert!(g.is_self_inverse(), "{:?} should be self-inverse", g);
    }
}

#[test]
fn test_clifford_gates() {
    let cliffords = [
        QuantumGate::H, QuantumGate::S, QuantumGate::Sdg,
        QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::SWAP,
    ];
    for g in &cliffords {
        assert!(g.is_clifford(), "{:?} should be Clifford", g);
    }
}

#[test]
fn test_non_clifford_gates() {
    let non_cliffords = [QuantumGate::T, QuantumGate::Tdg, QuantumGate::CCX];
    for g in &non_cliffords {
        assert!(!g.is_clifford(), "{:?} should not be Clifford", g);
    }
}

// ═══════════════════════════════════════════════════════════
// Edge cases: topology
// ═══════════════════════════════════════════════════════════

#[test]
fn test_topology_single_qubit() {
    let topo = DeviceTopology::linear(1);
    assert_eq!(topo.num_qubits, 1);
    assert_eq!(topo.edges.len(), 0);
    assert_eq!(topo.diameter(), 0);
}

#[test]
fn test_topology_two_qubits() {
    let topo = DeviceTopology::linear(2);
    assert!(topo.are_connected(0, 1));
    assert!(!topo.are_connected(0, 0)); // self-loop
}

#[test]
fn test_topology_grid_small() {
    let topo = DeviceTopology::grid(2, 2);
    assert_eq!(topo.num_qubits, 4);
    assert!(topo.are_connected(0, 1));
    assert!(topo.are_connected(0, 2));
    assert!(!topo.are_connected(0, 3)); // diagonal
}

#[test]
fn test_topology_are_connected_symmetric() {
    let topo = DeviceTopology::linear(5);
    for &(a, b) in &topo.edges {
        assert!(topo.are_connected(a, b));
        assert!(topo.are_connected(b, a));
    }
}

#[test]
fn test_topology_out_of_bounds() {
    let topo = DeviceTopology::linear(3);
    assert!(!topo.are_connected(0, 10));
    assert!(!topo.are_connected(99, 100));
}

// ═══════════════════════════════════════════════════════════
// Edge cases: Kraus
// ═══════════════════════════════════════════════════════════

#[test]
fn test_complex_matrix_zero() {
    let m = ComplexMatrix::new(2, 2);
    assert_eq!(m.get(0, 0), (0.0, 0.0));
    assert_eq!(m.get(1, 1), (0.0, 0.0));
}

#[test]
fn test_complex_matrix_1x1() {
    let mut m = ComplexMatrix::new(1, 1);
    m.set(0, 0, (3.0, 4.0));
    let tr = m.trace().unwrap();
    assert_eq!(tr, (3.0, 4.0));
}

#[test]
fn test_kraus_depolarizing_zero_error() {
    let ch = KrausChannel::depolarizing(0.0, 1);
    let f = ch.average_gate_fidelity();
    assert!((f - 1.0).abs() < 1e-6);
}

#[test]
fn test_kraus_depolarizing_max_error() {
    let ch = KrausChannel::depolarizing(1.0, 1);
    let f = ch.average_gate_fidelity();
    assert!(f >= 0.0 && f <= 1.0);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: QEC
// ═══════════════════════════════════════════════════════════

#[test]
fn test_qec_surface_code_d1() {
    let code = QecCode::SurfaceCode { distance: 1 };
    assert_eq!(code.physical_per_logical(), 1);
    assert_eq!(code.code_distance(), 1);
}

#[test]
fn test_qec_repetition_d1() {
    let code = QecCode::RepetitionCode { distance: 1 };
    assert_eq!(code.physical_per_logical(), 1);
}

#[test]
fn test_qec_ldpc_k_equals_n() {
    let code = QecCode::LdpcCode { n: 10, k: 10 };
    assert_eq!(code.physical_per_logical(), 1);
}

#[test]
fn test_qec_analysis_zero_rounds() {
    let analysis = QecAnalysis::analyse(1, 0, QecCode::SteaneCode, 0.001);
    assert!(analysis.logical_error_rate >= 0.0);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: cost model
// ═══════════════════════════════════════════════════════════

#[test]
fn test_cost_zero_flops() {
    let model = CostModel::a100();
    assert_eq!(model.compute_time_ms(0), 0.0);
}

#[test]
fn test_cost_zero_bytes() {
    let model = CostModel::a100();
    assert_eq!(model.memory_time_ms(0), 0.0);
}

#[test]
fn test_cost_arithmetic_intensity_zero_bytes() {
    let model = CostModel::a100();
    assert!(model.arithmetic_intensity(100, 0).is_infinite());
}

#[test]
fn test_cost_roofline_both_zero() {
    let model = CostModel::a100();
    assert_eq!(model.roofline_time_ms(0, 0), 0.0);
}

#[test]
fn test_quantum_cost_zero_gates() {
    let model = QuantumCostModel::superconducting_default();
    let f = model.circuit_fidelity(0, 0);
    assert_eq!(f, 1.0);
}

#[test]
fn test_quantum_cost_zero_time() {
    let model = QuantumCostModel::superconducting_default();
    let t = model.circuit_time_us(0, 0, 0, 0);
    assert_eq!(t, 0.0);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: budget
// ═══════════════════════════════════════════════════════════

#[test]
fn test_budget_all_none() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    assert!(budget.check_flops(u64::MAX).is_ok());
    assert!(budget.check_memory(u64::MAX).is_ok());
    assert!(budget.check_fidelity(0.0).is_ok());
}

#[test]
fn test_budget_exact_limit() {
    let budget = Budget {
        max_flops: Some(100),
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: None,
        max_circuit_depth: None,
    };
    assert!(budget.check_flops(100).is_ok());
    assert!(budget.check_flops(101).is_err());
}

#[test]
fn test_budget_fidelity_exact() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: Some(0.99),
        max_circuit_depth: None,
    };
    assert!(budget.check_fidelity(0.99).is_ok());
    assert!(budget.check_fidelity(0.989).is_err());
}

// ═══════════════════════════════════════════════════════════
// Edge cases: energy model
// ═══════════════════════════════════════════════════════════

#[test]
fn test_energy_zero_time() {
    let model = EnergyModel::a100();
    assert_eq!(model.energy_joules(0.0, 1), 0.0);
    assert_eq!(model.carbon_grams(0.0, 1), 0.0);
}

#[test]
fn test_energy_zero_gpus() {
    let model = EnergyModel::a100();
    let j = model.energy_joules(1000.0, 0);
    // Still has CPU + memory power
    assert!(j > 0.0);
}

#[test]
fn test_quantum_energy_zero_time() {
    let model = EnergyModel::a100();
    assert_eq!(model.quantum_energy_joules(0.0, 10), 0.0);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: IR printer
// ═══════════════════════════════════════════════════════════

#[test]
fn test_print_empty_context() {
    let ctx = Context::new();
    let output = print_ir(&ctx);
    assert!(output.is_empty() || output.trim().is_empty());
}

// ═══════════════════════════════════════════════════════════
// Edge cases: attributes
// ═══════════════════════════════════════════════════════════

#[test]
fn test_attributes_empty() {
    let attrs = Attributes::new();
    assert!(attrs.is_empty());
    assert_eq!(attrs.len(), 0);
    assert!(attrs.get("nonexistent").is_none());
}

#[test]
fn test_attributes_set_get() {
    let mut attrs = Attributes::new();
    attrs.set("x", Attribute::Integer(42));
    assert_eq!(attrs.get_integer("x"), Some(42));
    assert_eq!(attrs.get_float("x"), None);
}

#[test]
fn test_attributes_overwrite() {
    let mut attrs = Attributes::new();
    attrs.set("x", Attribute::Integer(1));
    attrs.set("x", Attribute::Integer(2));
    assert_eq!(attrs.get_integer("x"), Some(2));
    assert_eq!(attrs.len(), 1);
}

#[test]
fn test_attributes_remove() {
    let mut attrs = Attributes::new();
    attrs.set("a", Attribute::Bool(true));
    assert!(attrs.contains("a"));
    attrs.remove("a");
    assert!(!attrs.contains("a"));
    assert_eq!(attrs.len(), 0);
}

#[test]
fn test_attributes_mixed_types() {
    let mut attrs = Attributes::new();
    attrs.set("i", Attribute::Integer(10));
    attrs.set("f", Attribute::Float(3.14));
    attrs.set("b", Attribute::Bool(true));
    assert_eq!(attrs.get_integer("i"), Some(10));
    assert_eq!(attrs.get_float("f"), Some(3.14));
    assert_eq!(attrs.get_bool("b"), Some(true));
    assert_eq!(attrs.len(), 3);
}

#[test]
fn test_attributes_iter() {
    let mut attrs = Attributes::new();
    attrs.set("a", Attribute::Integer(1));
    attrs.set("b", Attribute::Integer(2));
    let count = attrs.iter().count();
    assert_eq!(count, 2);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: DataType
// ═══════════════════════════════════════════════════════════

#[test]
fn test_datatype_byte_sizes() {
    assert_eq!(DataType::FP64.byte_size(), 8);
    assert_eq!(DataType::FP32.byte_size(), 4);
    assert_eq!(DataType::FP16.byte_size(), 2);
    assert_eq!(DataType::BF16.byte_size(), 2);
    assert_eq!(DataType::INT8.byte_size(), 1);
}

#[test]
fn test_datatype_equality() {
    assert_eq!(DataType::FP32, DataType::FP32);
    assert_ne!(DataType::FP32, DataType::FP16);
}

// ═══════════════════════════════════════════════════════════
// Edge cases: Dimension
// ═══════════════════════════════════════════════════════════

#[test]
fn test_dimension_constant() {
    let d = Dimension::Constant(42);
    assert_eq!(d.static_value(), Some(42));
    assert!(d.is_static());
}

#[test]
fn test_dimension_symbolic() {
    let d = Dimension::Symbolic("batch".to_string());
    assert_eq!(d.static_value(), None);
    assert!(!d.is_static());
}

// ═══════════════════════════════════════════════════════════
// Edge cases: reactive budget cumulative
// ═══════════════════════════════════════════════════════════

#[test]
fn test_reactive_budget_cumulative_fidelity() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: None,
        min_fidelity: Some(0.5),
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    rb.consume(0, 0, 0.0, 0.9);
    rb.consume(0, 0, 0.0, 0.9);
    rb.consume(0, 0, 0.0, 0.9);
    // 0.9^3 = 0.729
    assert!((rb.current_fidelity - 0.729).abs() < 1e-6);
    assert!(rb.check_remaining().is_ok());

    rb.consume(0, 0, 0.0, 0.5);
    // 0.729 * 0.5 = 0.3645 < 0.5
    assert!(rb.check_remaining().is_err());
}

#[test]
fn test_reactive_budget_cumulative_time() {
    let budget = Budget {
        max_flops: None,
        max_memory_bytes: None,
        max_time_ms: Some(100.0),
        min_fidelity: None,
        max_circuit_depth: None,
    };
    let mut rb = ReactiveBudget::new(budget);
    for _ in 0..10 {
        rb.consume(0, 0, 10.0, 1.0);
    }
    assert!((rb.elapsed_ms - 100.0).abs() < 1e-10);
    assert!(rb.check_remaining().is_ok());
    rb.consume(0, 0, 0.001, 1.0);
    assert!(rb.check_remaining().is_err());
}
