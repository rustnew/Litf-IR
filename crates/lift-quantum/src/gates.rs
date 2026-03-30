use serde::{Serialize, Deserialize};

/// Hardware provider for native gate sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    IbmEagle,
    IbmKyoto,
    Rigetti,
    IonQ,
    Quantinuum,
    Simulator,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumGate {
    // ── Standard 1-qubit ──
    H, X, Y, Z, S, Sdg, T, Tdg, SX,
    // ── Parametric 1-qubit ──
    RX, RY, RZ, P, U1, U2, U3,
    // ── Fixed-angle 1-qubit ──
    Rx90, Rx180,
    // ── 2-qubit standard ──
    CX, CZ, CY, SWAP, ISWAP, ECR,
    RZX, XX, YY, ZZ,
    // ── Rigetti native ──
    CPhase, XY,
    // ── Controlled phase ──
    CP,
    // ── IonQ native ──
    GPI, GPI2, MS,
    // ── 3-qubit ──
    CCX, CSWAP,
    // ── Multi-controlled ──
    MCX, MCZ,
    // ── Special / control ──
    GlobalPhase,
    Delay,
    VirtualRZ,
    IfElse,
    // ── Measurement & control flow ──
    Measure, MeasureAll, Reset, Barrier, Init,
    ParamGate,
}

impl QuantumGate {
    pub fn op_name(&self) -> &'static str {
        match self {
            // 1-qubit standard
            Self::H => "quantum.h", Self::X => "quantum.x",
            Self::Y => "quantum.y", Self::Z => "quantum.z",
            Self::S => "quantum.s", Self::Sdg => "quantum.sdg",
            Self::T => "quantum.t", Self::Tdg => "quantum.tdg",
            Self::SX => "quantum.sx",
            // 1-qubit parametric
            Self::RX => "quantum.rx", Self::RY => "quantum.ry",
            Self::RZ => "quantum.rz", Self::P => "quantum.p",
            Self::U1 => "quantum.u1", Self::U2 => "quantum.u2",
            Self::U3 => "quantum.u3",
            // Fixed-angle
            Self::Rx90 => "quantum.rx90", Self::Rx180 => "quantum.rx180",
            // 2-qubit standard
            Self::CX => "quantum.cx", Self::CZ => "quantum.cz",
            Self::CY => "quantum.cy", Self::SWAP => "quantum.swap",
            Self::ISWAP => "quantum.iswap", Self::ECR => "quantum.ecr",
            Self::RZX => "quantum.rzx",
            Self::XX => "quantum.xx", Self::YY => "quantum.yy",
            Self::ZZ => "quantum.zz",
            // Rigetti
            Self::CPhase => "quantum.cphase", Self::XY => "quantum.xy",
            // Controlled phase
            Self::CP => "quantum.cp",
            // IonQ
            Self::GPI => "quantum.gpi", Self::GPI2 => "quantum.gpi2",
            Self::MS => "quantum.ms",
            // 3-qubit
            Self::CCX => "quantum.ccx", Self::CSWAP => "quantum.cswap",
            // Multi-controlled
            Self::MCX => "quantum.mcx", Self::MCZ => "quantum.mcz",
            // Special
            Self::GlobalPhase => "quantum.global_phase",
            Self::Delay => "quantum.delay",
            Self::VirtualRZ => "quantum.virtual_rz",
            Self::IfElse => "quantum.if_else",
            // Measurement / control
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
            "quantum.rx90" => Some(Self::Rx90), "quantum.rx180" => Some(Self::Rx180),
            "quantum.cx" => Some(Self::CX), "quantum.cz" => Some(Self::CZ),
            "quantum.cy" => Some(Self::CY), "quantum.swap" => Some(Self::SWAP),
            "quantum.iswap" => Some(Self::ISWAP), "quantum.ecr" => Some(Self::ECR),
            "quantum.rzx" => Some(Self::RZX),
            "quantum.xx" => Some(Self::XX), "quantum.yy" => Some(Self::YY),
            "quantum.zz" => Some(Self::ZZ),
            "quantum.cphase" => Some(Self::CPhase), "quantum.xy" => Some(Self::XY),
            "quantum.cp" => Some(Self::CP),
            "quantum.gpi" => Some(Self::GPI), "quantum.gpi2" => Some(Self::GPI2),
            "quantum.ms" => Some(Self::MS),
            "quantum.ccx" => Some(Self::CCX), "quantum.cswap" => Some(Self::CSWAP),
            "quantum.mcx" => Some(Self::MCX), "quantum.mcz" => Some(Self::MCZ),
            "quantum.global_phase" => Some(Self::GlobalPhase),
            "quantum.delay" => Some(Self::Delay),
            "quantum.virtual_rz" => Some(Self::VirtualRZ),
            "quantum.if_else" => Some(Self::IfElse),
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
            Self::Rx90 | Self::Rx180 |
            Self::GPI | Self::GPI2 | Self::VirtualRZ |
            Self::Measure | Self::Reset | Self::Init => 1,

            Self::CX | Self::CZ | Self::CY | Self::SWAP | Self::ISWAP |
            Self::ECR | Self::RZX | Self::XX | Self::YY | Self::ZZ |
            Self::CPhase | Self::XY | Self::CP | Self::MS => 2,

            Self::CCX | Self::CSWAP => 3,

            Self::MCX | Self::MCZ => 0, // variable, determined by attrs

            Self::GlobalPhase | Self::Delay | Self::IfElse |
            Self::MeasureAll | Self::Barrier | Self::ParamGate => 0,
        }
    }

    pub fn is_parametric(&self) -> bool {
        matches!(self,
            Self::RX | Self::RY | Self::RZ | Self::P |
            Self::U1 | Self::U2 | Self::U3 | Self::RZX |
            Self::XX | Self::YY | Self::ZZ |
            Self::CPhase | Self::XY | Self::CP |
            Self::GPI | Self::GPI2 | Self::MS |
            Self::GlobalPhase | Self::VirtualRZ |
            Self::ParamGate
        )
    }

    pub fn is_self_inverse(&self) -> bool {
        matches!(self,
            Self::H | Self::X | Self::Y | Self::Z |
            Self::CX | Self::CZ | Self::SWAP |
            Self::Rx180
        )
    }

    pub fn is_clifford(&self) -> bool {
        matches!(self,
            Self::H | Self::X | Self::Y | Self::Z | Self::S | Self::Sdg |
            Self::CX | Self::CZ | Self::SWAP | Self::ISWAP
        )
    }

    /// Returns the native basis set for a given hardware provider.
    pub fn native_basis(provider: Provider) -> &'static [QuantumGate] {
        match provider {
            Provider::IbmEagle | Provider::IbmKyoto => &[
                Self::RZ, Self::SX, Self::X, Self::CX, Self::ECR,
            ],
            Provider::Rigetti => &[
                Self::RZ, Self::RX, Self::CZ, Self::CPhase, Self::XY,
            ],
            Provider::IonQ => &[
                Self::GPI, Self::GPI2, Self::MS,
            ],
            Provider::Quantinuum => &[
                Self::RZ, Self::RX, Self::RY, Self::ZZ,
            ],
            Provider::Simulator => &[
                Self::H, Self::X, Self::Y, Self::Z, Self::S, Self::T,
                Self::RX, Self::RY, Self::RZ, Self::CX, Self::CZ,
                Self::CCX, Self::SWAP,
            ],
        }
    }

    /// Returns `true` if this gate is a measurement or classical control.
    #[inline]
    pub fn is_measurement(&self) -> bool {
        matches!(self, Self::Measure | Self::MeasureAll)
    }

    /// Returns `true` if this is a multi-qubit entangling gate.
    #[inline]
    pub fn is_entangling(&self) -> bool {
        matches!(self,
            Self::CX | Self::CZ | Self::CY | Self::SWAP | Self::ISWAP |
            Self::ECR | Self::RZX | Self::XX | Self::YY | Self::ZZ |
            Self::CPhase | Self::XY | Self::CP | Self::MS |
            Self::CCX | Self::CSWAP | Self::MCX | Self::MCZ
        )
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
