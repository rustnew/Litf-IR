use serde::{Serialize, Deserialize};

/// Ansatz type for variational quantum circuits.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnsatzType {
    HardwareEfficient,
    StronglyEntangling,
    TwoLocal,
    UCCSD,
    Custom,
}

/// Synchronisation policy for co-execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncPolicy {
    Blocking,
    Asynchronous,
    Pipeline,
}

/// Feature map for quantum kernel methods.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureMap {
    ZZFeatureMap,
    PauliFeatureMap,
    AngleEncoding,
    AmplitudeEncoding,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HybridOp {
    // ── Encoding / Decoding ──
    Encode,
    Decode,

    // ── Gradient methods ──
    ParameterShift,
    FiniteDifference,
    SPSA,
    AdjointDifferentiation,
    StochasticParameterShift,
    JointGradient,

    // ── Processing ──
    ClassicalPreprocess,
    QuantumPostprocess,
    HybridForward,
    HybridBackward,

    // ── Variational quantum circuits ──
    VqcLayer,
    VqeAnsatz,
    QaoaLayer,
    QuantumKernel,

    // ── Data transfer ──
    GpuToQpu,
    QpuToGpu,

    // ── Co-execution ──
    CoExecute,

    // ── Measurement basis ──
    MeasureExpectation,
    MeasureSamples,
}

impl HybridOp {
    pub fn op_name(&self) -> &'static str {
        match self {
            Self::Encode => "hybrid.encode",
            Self::Decode => "hybrid.decode",
            Self::ParameterShift => "hybrid.parameter_shift",
            Self::FiniteDifference => "hybrid.finite_difference",
            Self::SPSA => "hybrid.spsa",
            Self::AdjointDifferentiation => "hybrid.adjoint_diff",
            Self::StochasticParameterShift => "hybrid.stochastic_param_shift",
            Self::JointGradient => "hybrid.joint_gradient",
            Self::ClassicalPreprocess => "hybrid.classical_preprocess",
            Self::QuantumPostprocess => "hybrid.quantum_postprocess",
            Self::HybridForward => "hybrid.forward",
            Self::HybridBackward => "hybrid.backward",
            Self::VqcLayer => "hybrid.vqc_layer",
            Self::VqeAnsatz => "hybrid.vqe_ansatz",
            Self::QaoaLayer => "hybrid.qaoa_layer",
            Self::QuantumKernel => "hybrid.quantum_kernel",
            Self::GpuToQpu => "hybrid.gpu_to_qpu",
            Self::QpuToGpu => "hybrid.qpu_to_gpu",
            Self::CoExecute => "hybrid.co_execute",
            Self::MeasureExpectation => "hybrid.measure_expectation",
            Self::MeasureSamples => "hybrid.measure_samples",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "hybrid.encode" => Some(Self::Encode),
            "hybrid.decode" => Some(Self::Decode),
            "hybrid.parameter_shift" => Some(Self::ParameterShift),
            "hybrid.finite_difference" => Some(Self::FiniteDifference),
            "hybrid.spsa" => Some(Self::SPSA),
            "hybrid.adjoint_diff" => Some(Self::AdjointDifferentiation),
            "hybrid.stochastic_param_shift" => Some(Self::StochasticParameterShift),
            "hybrid.joint_gradient" => Some(Self::JointGradient),
            "hybrid.classical_preprocess" => Some(Self::ClassicalPreprocess),
            "hybrid.quantum_postprocess" => Some(Self::QuantumPostprocess),
            "hybrid.forward" => Some(Self::HybridForward),
            "hybrid.backward" => Some(Self::HybridBackward),
            "hybrid.vqc_layer" => Some(Self::VqcLayer),
            "hybrid.vqe_ansatz" => Some(Self::VqeAnsatz),
            "hybrid.qaoa_layer" => Some(Self::QaoaLayer),
            "hybrid.quantum_kernel" => Some(Self::QuantumKernel),
            "hybrid.gpu_to_qpu" => Some(Self::GpuToQpu),
            "hybrid.qpu_to_gpu" => Some(Self::QpuToGpu),
            "hybrid.co_execute" => Some(Self::CoExecute),
            "hybrid.measure_expectation" => Some(Self::MeasureExpectation),
            "hybrid.measure_samples" => Some(Self::MeasureSamples),
            _ => None,
        }
    }

    /// Returns `true` if this is a gradient computation operation.
    #[inline]
    pub fn is_gradient(&self) -> bool {
        matches!(self,
            Self::ParameterShift | Self::FiniteDifference | Self::SPSA |
            Self::AdjointDifferentiation | Self::StochasticParameterShift |
            Self::JointGradient
        )
    }

    /// Returns `true` if this is a variational algorithm operation.
    #[inline]
    pub fn is_variational(&self) -> bool {
        matches!(self,
            Self::VqcLayer | Self::VqeAnsatz | Self::QaoaLayer | Self::QuantumKernel
        )
    }
}
