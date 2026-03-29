use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NoiseModel {
    Ideal,
    Depolarizing { p: f64 },
    AmplitudeDamping { gamma: f64 },
    PhaseDamping { gamma: f64 },
    BitFlip { p: f64 },
    PhaseFlip { p: f64 },
    ThermalRelaxation { t1_us: f64, t2_us: f64, gate_time_us: f64 },
    Kraus { operators: Vec<Vec<(f64, f64)>> },
    Composed(Vec<NoiseModel>),
}

impl NoiseModel {
    pub fn fidelity(&self) -> f64 {
        match self {
            NoiseModel::Ideal => 1.0,
            NoiseModel::Depolarizing { p } => 1.0 - p,
            NoiseModel::AmplitudeDamping { gamma } => 1.0 - gamma / 2.0,
            NoiseModel::PhaseDamping { gamma } => 1.0 - gamma / 2.0,
            NoiseModel::BitFlip { p } => 1.0 - p,
            NoiseModel::PhaseFlip { p } => 1.0 - p,
            NoiseModel::ThermalRelaxation { t1_us, t2_us, gate_time_us } => {
                let p1 = (-gate_time_us / t1_us).exp();
                let p2 = (-gate_time_us / t2_us).exp();
                (1.0 + p1 + 2.0 * p2) / 4.0
            }
            NoiseModel::Kraus { .. } => 0.99, // approximate
            NoiseModel::Composed(models) => {
                models.iter().map(|m| m.fidelity()).product()
            }
        }
    }

    pub fn compose(&self, other: &NoiseModel) -> NoiseModel {
        match (self, other) {
            (NoiseModel::Ideal, m) | (m, NoiseModel::Ideal) => m.clone(),
            _ => NoiseModel::Composed(vec![self.clone(), other.clone()]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateNoise {
    pub gate_fidelity: f64,
    pub gate_time_us: f64,
    pub noise: NoiseModel,
}

impl GateNoise {
    pub fn ideal() -> Self {
        Self { gate_fidelity: 1.0, gate_time_us: 0.0, noise: NoiseModel::Ideal }
    }

    pub fn with_depolarizing(fidelity: f64, gate_time: f64) -> Self {
        Self {
            gate_fidelity: fidelity,
            gate_time_us: gate_time,
            noise: NoiseModel::Depolarizing { p: 1.0 - fidelity },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CircuitNoise {
    pub total_fidelity: f64,
    pub gate_count: usize,
    pub two_qubit_count: usize,
    pub circuit_depth: usize,
    pub total_time_us: f64,
}

impl CircuitNoise {
    pub fn new() -> Self {
        Self {
            total_fidelity: 1.0,
            gate_count: 0,
            two_qubit_count: 0,
            circuit_depth: 0,
            total_time_us: 0.0,
        }
    }

    pub fn add_gate(&mut self, noise: &GateNoise, is_two_qubit: bool) {
        self.total_fidelity *= noise.gate_fidelity;
        self.gate_count += 1;
        self.total_time_us += noise.gate_time_us;
        if is_two_qubit { self.two_qubit_count += 1; }
    }

    pub fn meets_threshold(&self, min_fidelity: f64) -> bool {
        self.total_fidelity >= min_fidelity
    }
}

impl Default for CircuitNoise {
    fn default() -> Self { Self::new() }
}
