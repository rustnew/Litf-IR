use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HybridOp {
    Encode,
    Decode,
    ParameterShift,
    FiniteDifference,
    SPSA,
    JointGradient,
    ClassicalPreprocess,
    QuantumPostprocess,
    HybridForward,
    HybridBackward,
    CoExecute,
}

impl HybridOp {
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::Encode => "hybrid.encode",
            Self::Decode => "hybrid.decode",
            Self::ParameterShift => "hybrid.parameter_shift",
            Self::FiniteDifference => "hybrid.finite_difference",
            Self::SPSA => "hybrid.spsa",
            Self::JointGradient => "hybrid.joint_gradient",
            Self::ClassicalPreprocess => "hybrid.classical_preprocess",
            Self::QuantumPostprocess => "hybrid.quantum_postprocess",
            Self::HybridForward => "hybrid.forward",
            Self::HybridBackward => "hybrid.backward",
            Self::CoExecute => "hybrid.co_execute",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "hybrid.encode" => Some(Self::Encode),
            "hybrid.decode" => Some(Self::Decode),
            "hybrid.parameter_shift" => Some(Self::ParameterShift),
            "hybrid.finite_difference" => Some(Self::FiniteDifference),
            "hybrid.spsa" => Some(Self::SPSA),
            "hybrid.joint_gradient" => Some(Self::JointGradient),
            "hybrid.classical_preprocess" => Some(Self::ClassicalPreprocess),
            "hybrid.quantum_postprocess" => Some(Self::QuantumPostprocess),
            "hybrid.forward" => Some(Self::HybridForward),
            "hybrid.backward" => Some(Self::HybridBackward),
            "hybrid.co_execute" => Some(Self::CoExecute),
            _ => None,
        }
    }
}
