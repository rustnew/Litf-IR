use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

// ═══════════════════════════════════════════════════════════
// EncodingStrategy name roundtrip
// ═══════════════════════════════════════════════════════════

#[test]
fn test_angle_encoding_name() {
    assert_eq!(EncodingStrategy::AngleEncoding.name(), "angle");
}

#[test]
fn test_amplitude_encoding_name() {
    assert_eq!(EncodingStrategy::AmplitudeEncoding.name(), "amplitude");
}

#[test]
fn test_basis_encoding_name() {
    assert_eq!(EncodingStrategy::BasisEncoding.name(), "basis");
}

#[test]
fn test_iqp_encoding_name() {
    assert_eq!(EncodingStrategy::IQPEncoding.name(), "iqp");
}

#[test]
fn test_hamiltonian_encoding_name() {
    assert_eq!(EncodingStrategy::HamiltonianEncoding.name(), "hamiltonian");
}

#[test]
fn test_kernel_encoding_name() {
    assert_eq!(EncodingStrategy::KernelEncoding.name(), "kernel");
}

// ═══════════════════════════════════════════════════════════
// Qubits required
// ═══════════════════════════════════════════════════════════

#[test]
fn test_angle_qubits_linear() {
    assert_eq!(EncodingStrategy::AngleEncoding.qubits_required(8), 8);
    assert_eq!(EncodingStrategy::AngleEncoding.qubits_required(1), 1);
    assert_eq!(EncodingStrategy::AngleEncoding.qubits_required(100), 100);
}

#[test]
fn test_amplitude_qubits_logarithmic() {
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(1), 1);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(2), 1);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(4), 2);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(8), 3);
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(256), 8);
}

#[test]
fn test_amplitude_qubits_non_power_of_two() {
    // ceil(log2(5)) = 3
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(5), 3);
    // ceil(log2(3)) = 2
    assert_eq!(EncodingStrategy::AmplitudeEncoding.qubits_required(3), 2);
}

#[test]
fn test_basis_qubits() {
    assert_eq!(EncodingStrategy::BasisEncoding.qubits_required(10), 10);
}

#[test]
fn test_iqp_qubits() {
    assert_eq!(EncodingStrategy::IQPEncoding.qubits_required(6), 6);
}

#[test]
fn test_hamiltonian_qubits() {
    assert_eq!(EncodingStrategy::HamiltonianEncoding.qubits_required(4), 4);
}

#[test]
fn test_kernel_qubits() {
    assert_eq!(EncodingStrategy::KernelEncoding.qubits_required(12), 12);
}

// ═══════════════════════════════════════════════════════════
// Circuit depth
// ═══════════════════════════════════════════════════════════

#[test]
fn test_angle_depth_constant() {
    assert_eq!(EncodingStrategy::AngleEncoding.circuit_depth(1), 1);
    assert_eq!(EncodingStrategy::AngleEncoding.circuit_depth(100), 1);
}

#[test]
fn test_amplitude_depth_linear() {
    assert_eq!(EncodingStrategy::AmplitudeEncoding.circuit_depth(8), 8);
}

#[test]
fn test_basis_depth_constant() {
    assert_eq!(EncodingStrategy::BasisEncoding.circuit_depth(50), 1);
}

#[test]
fn test_iqp_depth_quadratic_like() {
    assert_eq!(EncodingStrategy::IQPEncoding.circuit_depth(4), 8);
    assert_eq!(EncodingStrategy::IQPEncoding.circuit_depth(10), 20);
}

#[test]
fn test_hamiltonian_depth() {
    assert_eq!(EncodingStrategy::HamiltonianEncoding.circuit_depth(5), 5);
}

#[test]
fn test_kernel_depth() {
    assert_eq!(EncodingStrategy::KernelEncoding.circuit_depth(4), 12);
}

// ═══════════════════════════════════════════════════════════
// EncodingConfig
// ═══════════════════════════════════════════════════════════

#[test]
fn test_encoding_config_new() {
    let config = EncodingConfig::new(EncodingStrategy::AngleEncoding, 10);
    assert_eq!(config.strategy, EncodingStrategy::AngleEncoding);
    assert_eq!(config.classical_dim, 10);
    assert_eq!(config.num_qubits, 10);
    assert_eq!(config.repetitions, 1);
}

#[test]
fn test_encoding_config_amplitude() {
    let config = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 16);
    assert_eq!(config.num_qubits, 4); // log2(16) = 4
}

#[test]
fn test_encoding_config_equality() {
    let c1 = EncodingConfig::new(EncodingStrategy::AngleEncoding, 8);
    let c2 = EncodingConfig::new(EncodingStrategy::AngleEncoding, 8);
    assert_eq!(c1, c2);
}

#[test]
fn test_encoding_config_inequality() {
    let c1 = EncodingConfig::new(EncodingStrategy::AngleEncoding, 8);
    let c2 = EncodingConfig::new(EncodingStrategy::BasisEncoding, 8);
    assert_ne!(c1, c2);
}

#[test]
fn test_encoding_strategy_clone() {
    let s = EncodingStrategy::IQPEncoding;
    let s2 = s.clone();
    assert_eq!(s, s2);
}

#[test]
fn test_encoding_strategy_debug() {
    let s = EncodingStrategy::KernelEncoding;
    let dbg = format!("{:?}", s);
    assert!(dbg.contains("Kernel"));
}
