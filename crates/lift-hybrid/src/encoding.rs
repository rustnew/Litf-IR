use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncodingStrategy {
    AngleEncoding,
    AmplitudeEncoding,
    BasisEncoding,
    IQPEncoding,
    HamiltonianEncoding,
    KernelEncoding,
}

impl EncodingStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::AngleEncoding => "angle",
            Self::AmplitudeEncoding => "amplitude",
            Self::BasisEncoding => "basis",
            Self::IQPEncoding => "iqp",
            Self::HamiltonianEncoding => "hamiltonian",
            Self::KernelEncoding => "kernel",
        }
    }

    pub fn qubits_required(&self, classical_dim: usize) -> usize {
        match self {
            Self::AngleEncoding => classical_dim,
            Self::AmplitudeEncoding => {
                // ceil(log2(classical_dim))
                if classical_dim <= 1 { 1 }
                else { (classical_dim as f64).log2().ceil() as usize }
            }
            Self::BasisEncoding => classical_dim,
            Self::IQPEncoding => classical_dim,
            Self::HamiltonianEncoding => classical_dim,
            Self::KernelEncoding => classical_dim,
        }
    }

    pub fn circuit_depth(&self, classical_dim: usize) -> usize {
        match self {
            Self::AngleEncoding => 1,
            Self::AmplitudeEncoding => classical_dim,
            Self::BasisEncoding => 1,
            Self::IQPEncoding => classical_dim * 2,
            Self::HamiltonianEncoding => classical_dim,
            Self::KernelEncoding => classical_dim * 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub strategy: EncodingStrategy,
    pub classical_dim: usize,
    pub num_qubits: usize,
    pub repetitions: usize,
}

impl EncodingConfig {
    pub fn new(strategy: EncodingStrategy, classical_dim: usize) -> Self {
        let num_qubits = strategy.qubits_required(classical_dim);
        Self {
            strategy,
            classical_dim,
            num_qubits,
            repetitions: 1,
        }
    }
}
