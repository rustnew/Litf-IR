use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumType {
    Qubit,
    PhysicalQubit {
        id: usize,
        t1_us: f64,
        t2_us: f64,
        freq_ghz: f64,
        fidelity: f64,
    },
    ClassicalBit,
    QuantumState {
        dimension: usize,
        representation: StateRepr,
    },
    Hamiltonian {
        num_qubits: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateRepr {
    StateVector,
    DensityMatrix,
    MPS,
    Stabiliser,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PauliTerm {
    pub coeff: f64,
    pub ops: Vec<PauliOp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PauliOp {
    I,
    X,
    Y,
    Z,
}

impl QuantumType {
    pub fn is_linear(&self) -> bool {
        matches!(self, QuantumType::Qubit | QuantumType::PhysicalQubit { .. })
    }
}
