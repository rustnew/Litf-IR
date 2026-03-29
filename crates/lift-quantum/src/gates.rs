use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumGate {
    H, X, Y, Z, S, Sdg, T, Tdg, SX,
    RX, RY, RZ, P, U1, U2, U3,
    CX, CZ, CY, SWAP, ISWAP, ECR,
    RZX, XX, YY, ZZ,
    CCX, CSWAP,
    Measure, MeasureAll, Reset, Barrier, Init,
    ParamGate,
}

impl QuantumGate {
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::H => "quantum.h", Self::X => "quantum.x",
            Self::Y => "quantum.y", Self::Z => "quantum.z",
            Self::S => "quantum.s", Self::Sdg => "quantum.sdg",
            Self::T => "quantum.t", Self::Tdg => "quantum.tdg",
            Self::SX => "quantum.sx",
            Self::RX => "quantum.rx", Self::RY => "quantum.ry",
            Self::RZ => "quantum.rz", Self::P => "quantum.p",
            Self::U1 => "quantum.u1", Self::U2 => "quantum.u2",
            Self::U3 => "quantum.u3",
            Self::CX => "quantum.cx", Self::CZ => "quantum.cz",
            Self::CY => "quantum.cy", Self::SWAP => "quantum.swap",
            Self::ISWAP => "quantum.iswap", Self::ECR => "quantum.ecr",
            Self::RZX => "quantum.rzx",
            Self::XX => "quantum.xx", Self::YY => "quantum.yy",
            Self::ZZ => "quantum.zz",
            Self::CCX => "quantum.ccx", Self::CSWAP => "quantum.cswap",
            Self::Measure => "quantum.measure",
            Self::MeasureAll => "quantum.measure_all",
            Self::Reset => "quantum.reset",
            Self::Barrier => "quantum.barrier",
            Self::Init => "quantum.init",
            Self::ParamGate => "quantum.param_gate",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "quantum.h" => Some(Self::H), "quantum.x" => Some(Self::X),
            "quantum.y" => Some(Self::Y), "quantum.z" => Some(Self::Z),
            "quantum.s" => Some(Self::S), "quantum.sdg" => Some(Self::Sdg),
            "quantum.t" => Some(Self::T), "quantum.tdg" => Some(Self::Tdg),
            "quantum.sx" => Some(Self::SX),
            "quantum.rx" => Some(Self::RX), "quantum.ry" => Some(Self::RY),
            "quantum.rz" => Some(Self::RZ), "quantum.p" => Some(Self::P),
            "quantum.u1" => Some(Self::U1), "quantum.u2" => Some(Self::U2),
            "quantum.u3" => Some(Self::U3),
            "quantum.cx" => Some(Self::CX), "quantum.cz" => Some(Self::CZ),
            "quantum.cy" => Some(Self::CY), "quantum.swap" => Some(Self::SWAP),
            "quantum.iswap" => Some(Self::ISWAP), "quantum.ecr" => Some(Self::ECR),
            "quantum.rzx" => Some(Self::RZX),
            "quantum.xx" => Some(Self::XX), "quantum.yy" => Some(Self::YY),
            "quantum.zz" => Some(Self::ZZ),
            "quantum.ccx" => Some(Self::CCX), "quantum.cswap" => Some(Self::CSWAP),
            "quantum.measure" => Some(Self::Measure),
            "quantum.measure_all" => Some(Self::MeasureAll),
            "quantum.reset" => Some(Self::Reset),
            "quantum.barrier" => Some(Self::Barrier),
            "quantum.init" => Some(Self::Init),
            "quantum.param_gate" => Some(Self::ParamGate),
            _ => None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        match self {
            Self::H | Self::X | Self::Y | Self::Z | Self::S | Self::Sdg |
            Self::T | Self::Tdg | Self::SX | Self::RX | Self::RY | Self::RZ |
            Self::P | Self::U1 | Self::U2 | Self::U3 |
            Self::Measure | Self::Reset | Self::Init => 1,
            Self::CX | Self::CZ | Self::CY | Self::SWAP | Self::ISWAP |
            Self::ECR | Self::RZX | Self::XX | Self::YY | Self::ZZ => 2,
            Self::CCX | Self::CSWAP => 3,
            Self::MeasureAll | Self::Barrier | Self::ParamGate => 0,
        }
    }

    pub fn is_parametric(&self) -> bool {
        matches!(self, Self::RX | Self::RY | Self::RZ | Self::P |
                 Self::U1 | Self::U2 | Self::U3 | Self::RZX |
                 Self::XX | Self::YY | Self::ZZ | Self::ParamGate)
    }

    pub fn is_self_inverse(&self) -> bool {
        matches!(self, Self::H | Self::X | Self::Y | Self::Z | Self::CX | Self::CZ | Self::SWAP)
    }

    pub fn is_clifford(&self) -> bool {
        matches!(self, Self::H | Self::X | Self::Y | Self::Z | Self::S | Self::Sdg |
                 Self::CX | Self::CZ | Self::SWAP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_roundtrip() {
        for gate in &[QuantumGate::H, QuantumGate::CX, QuantumGate::RZ, QuantumGate::CCX] {
            assert_eq!(QuantumGate::from_name(gate.op_name()).as_ref(), Some(gate));
        }
    }

    #[test]
    fn test_qubit_counts() {
        assert_eq!(QuantumGate::H.num_qubits(), 1);
        assert_eq!(QuantumGate::CX.num_qubits(), 2);
        assert_eq!(QuantumGate::CCX.num_qubits(), 3);
    }
}
