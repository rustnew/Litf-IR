use lift_quantum::gates::{QuantumGate, Provider};
use lift_quantum::kraus::{ComplexMatrix, KrausChannel};
use lift_quantum::qec::{QecCode, QecAnalysis};
use lift_quantum::topology::DeviceTopology;
use lift_quantum::noise::{NoiseModel, GateNoise, CircuitNoise};

// ═══════════════════════════════════════════════════════════
// Gate tests — new gates
// ═══════════════════════════════════════════════════════════

#[test]
fn test_new_gate_roundtrip() {
    let gates = [
        QuantumGate::Rx90, QuantumGate::Rx180,
        QuantumGate::CPhase, QuantumGate::XY, QuantumGate::CP,
        QuantumGate::GPI, QuantumGate::GPI2, QuantumGate::MS,
        QuantumGate::MCX, QuantumGate::MCZ,
        QuantumGate::GlobalPhase, QuantumGate::Delay,
        QuantumGate::VirtualRZ, QuantumGate::IfElse,
    ];
    for gate in &gates {
        let name = gate.op_name();
        let recovered = QuantumGate::from_name(name);
        assert_eq!(recovered.as_ref(), Some(gate), "roundtrip failed for {:?}", gate);
    }
}

#[test]
fn test_rx90_properties() {
    let g = QuantumGate::Rx90;
    assert_eq!(g.num_qubits(), 1);
    assert!(!g.is_self_inverse());
    assert!(!g.is_clifford());
}

#[test]
fn test_rx180_self_inverse() {
    assert!(QuantumGate::Rx180.is_self_inverse());
}

#[test]
fn test_ionq_gates() {
    assert_eq!(QuantumGate::GPI.num_qubits(), 1);
    assert_eq!(QuantumGate::GPI2.num_qubits(), 1);
    assert_eq!(QuantumGate::MS.num_qubits(), 2);
    assert!(QuantumGate::GPI.is_parametric());
    assert!(QuantumGate::GPI2.is_parametric());
    assert!(QuantumGate::MS.is_parametric());
}

#[test]
fn test_rigetti_gates() {
    assert_eq!(QuantumGate::CPhase.num_qubits(), 2);
    assert_eq!(QuantumGate::XY.num_qubits(), 2);
    assert!(QuantumGate::CPhase.is_parametric());
    assert!(QuantumGate::XY.is_parametric());
}

#[test]
fn test_multi_controlled() {
    assert_eq!(QuantumGate::MCX.num_qubits(), 0); // variable
    assert_eq!(QuantumGate::MCZ.num_qubits(), 0);
    assert!(QuantumGate::MCX.is_entangling());
    assert!(QuantumGate::MCZ.is_entangling());
}

#[test]
fn test_special_gates() {
    assert!(QuantumGate::GlobalPhase.is_parametric());
    assert!(QuantumGate::VirtualRZ.is_parametric());
    assert_eq!(QuantumGate::VirtualRZ.num_qubits(), 1);
    assert_eq!(QuantumGate::Delay.num_qubits(), 0);
}

#[test]
fn test_entangling_classification() {
    let entangling = [
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::CY,
        QuantumGate::SWAP, QuantumGate::ISWAP, QuantumGate::ECR,
        QuantumGate::CPhase, QuantumGate::XY, QuantumGate::CP,
        QuantumGate::MS, QuantumGate::CCX, QuantumGate::CSWAP,
    ];
    for g in &entangling {
        assert!(g.is_entangling(), "{:?} should be entangling", g);
    }

    let non_entangling = [
        QuantumGate::H, QuantumGate::X, QuantumGate::RZ,
        QuantumGate::Measure, QuantumGate::Barrier,
    ];
    for g in &non_entangling {
        assert!(!g.is_entangling(), "{:?} should not be entangling", g);
    }
}

#[test]
fn test_measurement_classification() {
    assert!(QuantumGate::Measure.is_measurement());
    assert!(QuantumGate::MeasureAll.is_measurement());
    assert!(!QuantumGate::H.is_measurement());
    assert!(!QuantumGate::CX.is_measurement());
}

// ═══════════════════════════════════════════════════════════
// Provider / native basis tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_ibm_native_basis() {
    let basis = QuantumGate::native_basis(Provider::IbmEagle);
    assert!(basis.contains(&QuantumGate::RZ));
    assert!(basis.contains(&QuantumGate::SX));
    assert!(basis.contains(&QuantumGate::CX));
    assert!(basis.contains(&QuantumGate::ECR));
}

#[test]
fn test_rigetti_native_basis() {
    let basis = QuantumGate::native_basis(Provider::Rigetti);
    assert!(basis.contains(&QuantumGate::RZ));
    assert!(basis.contains(&QuantumGate::RX));
    assert!(basis.contains(&QuantumGate::CZ));
    assert!(basis.contains(&QuantumGate::CPhase));
    assert!(basis.contains(&QuantumGate::XY));
}

#[test]
fn test_ionq_native_basis() {
    let basis = QuantumGate::native_basis(Provider::IonQ);
    assert_eq!(basis.len(), 3);
    assert!(basis.contains(&QuantumGate::GPI));
    assert!(basis.contains(&QuantumGate::GPI2));
    assert!(basis.contains(&QuantumGate::MS));
}

#[test]
fn test_quantinuum_native_basis() {
    let basis = QuantumGate::native_basis(Provider::Quantinuum);
    assert!(basis.contains(&QuantumGate::ZZ));
}

#[test]
fn test_simulator_basis_is_universal() {
    let basis = QuantumGate::native_basis(Provider::Simulator);
    assert!(basis.len() >= 10);
    assert!(basis.contains(&QuantumGate::H));
    assert!(basis.contains(&QuantumGate::CCX));
}

// ═══════════════════════════════════════════════════════════
// ComplexMatrix tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_identity_matrix() {
    let id = ComplexMatrix::identity(3);
    assert_eq!(id.rows, 3);
    assert_eq!(id.cols, 3);
    assert_eq!(id.get(0, 0), (1.0, 0.0));
    assert_eq!(id.get(1, 1), (1.0, 0.0));
    assert_eq!(id.get(2, 2), (1.0, 0.0));
    assert_eq!(id.get(0, 1), (0.0, 0.0));
}

#[test]
fn test_matrix_multiply_identity() {
    let id = ComplexMatrix::identity(2);
    let mut m = ComplexMatrix::new(2, 2);
    m.set(0, 0, (1.0, 2.0));
    m.set(0, 1, (3.0, 4.0));
    m.set(1, 0, (5.0, 6.0));
    m.set(1, 1, (7.0, 8.0));
    let result = id.mul(&m).unwrap();
    assert_eq!(result.get(0, 0), (1.0, 2.0));
    assert_eq!(result.get(1, 1), (7.0, 8.0));
}

#[test]
fn test_matrix_incompatible_dims() {
    let a = ComplexMatrix::new(2, 3);
    let b = ComplexMatrix::new(4, 2);
    assert!(a.mul(&b).is_none());
}

#[test]
fn test_matrix_trace() {
    let mut m = ComplexMatrix::new(2, 2);
    m.set(0, 0, (3.0, 1.0));
    m.set(1, 1, (5.0, -2.0));
    let tr = m.trace().unwrap();
    assert!((tr.0 - 8.0).abs() < 1e-10);
    assert!((tr.1 - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_non_square_trace() {
    let m = ComplexMatrix::new(2, 3);
    assert!(m.trace().is_none());
}

#[test]
fn test_dagger_hermitian() {
    // Pauli X is hermitian
    let mut x = ComplexMatrix::new(2, 2);
    x.set(0, 1, (1.0, 0.0));
    x.set(1, 0, (1.0, 0.0));
    let xd = x.dagger();
    assert_eq!(xd.get(0, 1), (1.0, 0.0));
    assert_eq!(xd.get(1, 0), (1.0, 0.0));
}

#[test]
fn test_dagger_anti_hermitian() {
    // Pauli Y has imaginary off-diagonals
    let mut y = ComplexMatrix::new(2, 2);
    y.set(0, 1, (0.0, -1.0));
    y.set(1, 0, (0.0, 1.0));
    let yd = y.dagger();
    assert_eq!(yd.get(0, 1), (0.0, -1.0));
    assert_eq!(yd.get(1, 0), (0.0, 1.0));
}

// ═══════════════════════════════════════════════════════════
// Kraus channel tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_kraus_empty_operators() {
    assert!(KrausChannel::new(vec![]).is_none());
}

#[test]
fn test_kraus_inconsistent_dims() {
    let a = ComplexMatrix::new(2, 2);
    let b = ComplexMatrix::new(3, 3);
    assert!(KrausChannel::new(vec![a, b]).is_none());
}

#[test]
fn test_identity_channel() {
    let ch = KrausChannel::new(vec![ComplexMatrix::identity(2)]).unwrap();
    assert_eq!(ch.dimension, 2);
    let f = ch.average_gate_fidelity();
    assert!((f - 1.0).abs() < 1e-10);
}

#[test]
fn test_depolarizing_fidelity_scales() {
    let ch_low = KrausChannel::depolarizing(0.01, 1);
    let ch_high = KrausChannel::depolarizing(0.1, 1);
    assert!(ch_low.average_gate_fidelity() > ch_high.average_gate_fidelity());
}

#[test]
fn test_amplitude_damping_zero() {
    let ch = KrausChannel::amplitude_damping(0.0);
    let f = ch.average_gate_fidelity();
    assert!((f - 1.0).abs() < 1e-6);
}

#[test]
fn test_phase_damping_zero() {
    let ch = KrausChannel::phase_damping(0.0);
    let f = ch.average_gate_fidelity();
    assert!((f - 1.0).abs() < 1e-6);
}

#[test]
fn test_pauli_channel_symmetric() {
    let ch = KrausChannel::pauli(0.01, 0.01, 0.01);
    assert_eq!(ch.dimension, 2);
    assert_eq!(ch.operators.len(), 4);
}

#[test]
fn test_compose_dimension_mismatch() {
    let c1 = KrausChannel::depolarizing(0.01, 1); // dim=2
    let c2 = KrausChannel::new(vec![ComplexMatrix::identity(4)]).unwrap(); // dim=4
    assert!(c1.compose(&c2).is_none());
}

#[test]
fn test_compose_preserves_dimension() {
    let c1 = KrausChannel::amplitude_damping(0.01);
    let c2 = KrausChannel::phase_damping(0.01);
    let composed = c1.compose(&c2).unwrap();
    assert_eq!(composed.dimension, 2);
}

#[test]
fn test_compose_lowers_fidelity() {
    let c1 = KrausChannel::depolarizing(0.05, 1);
    let c2 = KrausChannel::amplitude_damping(0.05);
    let composed = c1.compose(&c2).unwrap();
    assert!(composed.average_gate_fidelity() <= c1.average_gate_fidelity());
}

// ═══════════════════════════════════════════════════════════
// QEC tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_surface_code_properties() {
    let code = QecCode::SurfaceCode { distance: 3 };
    assert_eq!(code.physical_per_logical(), 9);
    assert_eq!(code.code_distance(), 3);
    assert_eq!(code.syndrome_circuit_depth(), 3);
}

#[test]
fn test_surface_code_d5() {
    let code = QecCode::SurfaceCode { distance: 5 };
    assert_eq!(code.physical_per_logical(), 25);
    assert_eq!(code.code_distance(), 5);
}

#[test]
fn test_steane_code_properties() {
    let code = QecCode::SteaneCode;
    assert_eq!(code.physical_per_logical(), 7);
    assert_eq!(code.code_distance(), 3);
    assert_eq!(code.syndrome_circuit_depth(), 4);
}

#[test]
fn test_shor_code_properties() {
    let code = QecCode::ShorCode;
    assert_eq!(code.physical_per_logical(), 9);
    assert_eq!(code.code_distance(), 3);
    assert_eq!(code.syndrome_circuit_depth(), 8);
}

#[test]
fn test_repetition_code() {
    let code = QecCode::RepetitionCode { distance: 7 };
    assert_eq!(code.physical_per_logical(), 7);
    assert_eq!(code.code_distance(), 7);
    assert_eq!(code.syndrome_circuit_depth(), 2);
}

#[test]
fn test_ldpc_code() {
    let code = QecCode::LdpcCode { n: 100, k: 10 };
    assert_eq!(code.physical_per_logical(), 10);
}

#[test]
fn test_ldpc_code_zero_k() {
    let code = QecCode::LdpcCode { n: 50, k: 0 };
    assert_eq!(code.physical_per_logical(), 50);
}

#[test]
fn test_qec_analysis_physical_qubits() {
    let analysis = QecAnalysis::analyse(
        10, 100,
        QecCode::SurfaceCode { distance: 5 },
        0.001,
    );
    assert_eq!(analysis.physical_qubits, 250); // 10 * 25
    assert_eq!(analysis.logical_qubits, 10);
    assert_eq!(analysis.overhead_qubits, 25);
}

#[test]
fn test_qec_error_rate_valid() {
    let analysis = QecAnalysis::analyse(
        5, 50,
        QecCode::SurfaceCode { distance: 7 },
        0.001,
    );
    assert!(analysis.logical_error_rate >= 0.0);
    assert!(analysis.logical_error_rate <= 1.0);
}

#[test]
fn test_qec_higher_distance_lower_error() {
    let a1 = QecAnalysis::analyse(5, 50, QecCode::SurfaceCode { distance: 5 }, 0.001);
    let a2 = QecAnalysis::analyse(5, 50, QecCode::SurfaceCode { distance: 11 }, 0.001);
    assert!(a2.logical_error_rate <= a1.logical_error_rate);
}

#[test]
fn test_qec_meets_target() {
    let analysis = QecAnalysis::analyse(
        1, 10,
        QecCode::SurfaceCode { distance: 15 },
        0.001,
    );
    assert!(analysis.meets_target(0.1));
}

#[test]
fn test_suggest_distance_returns_odd() {
    let d = QecAnalysis::suggest_distance(0.001, 1e-6, 100);
    assert!(d >= 3);
    assert!(d % 2 == 1 || d == 0 || d == 51);
}

#[test]
fn test_suggest_distance_above_threshold() {
    // phys_error_rate >= threshold means QEC can't help
    let d = QecAnalysis::suggest_distance(0.02, 1e-6, 100);
    assert_eq!(d, 0);
}

// ═══════════════════════════════════════════════════════════
// Topology tests — new constructors
// ═══════════════════════════════════════════════════════════

#[test]
fn test_heavy_hex_topology() {
    let topo = DeviceTopology::heavy_hex(12);
    assert_eq!(topo.num_qubits, 12);
    assert!(topo.are_connected(0, 1));
    assert!(topo.are_connected(0, 3)); // cross-link
}

#[test]
fn test_all_to_all_topology() {
    let topo = DeviceTopology::all_to_all(5);
    assert_eq!(topo.num_qubits, 5);
    for i in 0..5 {
        for j in (i + 1)..5 {
            assert!(topo.are_connected(i, j));
        }
    }
    assert_eq!(topo.edges.len(), 10); // C(5,2) = 10
}

#[test]
fn test_tree_topology() {
    let topo = DeviceTopology::tree(7);
    assert_eq!(topo.num_qubits, 7);
    assert!(topo.are_connected(0, 1)); // root -> left
    assert!(topo.are_connected(0, 2)); // root -> right
    assert!(topo.are_connected(1, 3));
    assert!(topo.are_connected(1, 4));
    assert!(!topo.are_connected(1, 2)); // siblings not connected
}

#[test]
fn test_custom_topology() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    let topo = DeviceTopology::custom("ring4", &edges, 0.99);
    assert_eq!(topo.num_qubits, 4);
    assert!(topo.are_connected(0, 3));
}

#[test]
fn test_avg_connectivity() {
    let topo = DeviceTopology::all_to_all(4);
    let avg = topo.avg_connectivity();
    assert!((avg - 3.0).abs() < 1e-10); // each node connected to 3 others
}

#[test]
fn test_diameter_linear() {
    let topo = DeviceTopology::linear(5);
    assert_eq!(topo.diameter(), 4);
}

#[test]
fn test_diameter_all_to_all() {
    let topo = DeviceTopology::all_to_all(5);
    assert_eq!(topo.diameter(), 1);
}

#[test]
fn test_empty_topology() {
    let topo = DeviceTopology::new("empty", 0);
    assert_eq!(topo.avg_connectivity(), 0.0);
    assert_eq!(topo.diameter(), 0);
}

#[test]
fn test_shortest_path_tree() {
    let topo = DeviceTopology::tree(7);
    let path = topo.shortest_path(3, 4);
    assert!(path.is_some());
    let p = path.unwrap();
    assert!(p.len() <= 3); // 3 -> 1 -> 4
}

// ═══════════════════════════════════════════════════════════
// Noise model tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_noise_compose_with_ideal() {
    let dep = NoiseModel::Depolarizing { p: 0.01 };
    let composed = dep.compose(&NoiseModel::Ideal);
    assert_eq!(composed.fidelity(), dep.fidelity());
}

#[test]
fn test_noise_compose_two() {
    let n1 = NoiseModel::Depolarizing { p: 0.01 };
    let n2 = NoiseModel::BitFlip { p: 0.02 };
    let composed = n1.compose(&n2);
    let expected = n1.fidelity() * n2.fidelity();
    assert!((composed.fidelity() - expected).abs() < 1e-10);
}

#[test]
fn test_gate_noise_ideal() {
    let gn = GateNoise::ideal();
    assert_eq!(gn.gate_fidelity, 1.0);
    assert_eq!(gn.gate_time_us, 0.0);
}

#[test]
fn test_circuit_noise_accumulation() {
    let mut cn = CircuitNoise::new();
    let gn1 = GateNoise::with_depolarizing(0.999, 0.02);
    let gn2 = GateNoise::with_depolarizing(0.99, 0.3);
    cn.add_gate(&gn1, false);
    cn.add_gate(&gn2, true);
    assert_eq!(cn.gate_count, 2);
    assert_eq!(cn.two_qubit_count, 1);
    assert!(cn.total_fidelity < 1.0);
    assert!(cn.total_time_us > 0.0);
}

#[test]
fn test_circuit_noise_threshold() {
    let mut cn = CircuitNoise::new();
    assert!(cn.meets_threshold(0.99));
    let gn = GateNoise::with_depolarizing(0.5, 1.0);
    cn.add_gate(&gn, false);
    assert!(!cn.meets_threshold(0.99));
}

#[test]
fn test_thermal_relaxation_fidelity() {
    let model = NoiseModel::ThermalRelaxation {
        t1_us: 100.0,
        t2_us: 80.0,
        gate_time_us: 0.3,
    };
    let f = model.fidelity();
    assert!(f > 0.99 && f <= 1.0);
}

#[test]
fn test_phase_flip_fidelity() {
    let model = NoiseModel::PhaseFlip { p: 0.05 };
    assert!((model.fidelity() - 0.95).abs() < 1e-10);
}

#[test]
fn test_amplitude_damping_fidelity() {
    let model = NoiseModel::AmplitudeDamping { gamma: 0.1 };
    assert!((model.fidelity() - 0.95).abs() < 1e-10);
}
