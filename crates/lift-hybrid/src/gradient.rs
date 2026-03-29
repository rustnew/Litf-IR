use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GradientMethod {
    ParameterShift,
    FiniteDifference,
    SPSA,
    Adjoint,
    Backprop,
}

impl GradientMethod {
    pub fn circuit_evaluations(&self, num_params: usize) -> usize {
        match self {
            Self::ParameterShift => 2 * num_params,
            Self::FiniteDifference => num_params + 1,
            Self::SPSA => 2,
            Self::Adjoint => 1,
            Self::Backprop => 1,
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Self::ParameterShift | Self::Adjoint | Self::Backprop)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointGradientConfig {
    pub classical_method: GradientMethod,
    pub quantum_method: GradientMethod,
    pub num_classical_params: usize,
    pub num_quantum_params: usize,
}

impl JointGradientConfig {
    pub fn total_evaluations(&self) -> usize {
        self.classical_method.circuit_evaluations(self.num_classical_params)
            + self.quantum_method.circuit_evaluations(self.num_quantum_params)
    }
}
