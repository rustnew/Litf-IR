use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LithConfig {
    pub target: TargetConfig,
    pub budget: BudgetConfig,
    pub optimisation: OptimisationConfig,
    pub simulation: SimulationConfig,
    pub quantum: Option<QuantumConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetConfig {
    pub backend: String,
    pub device: Option<String>,
    pub precision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub max_flops: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_time_ms: Option<f64>,
    pub min_fidelity: Option<f64>,
    pub max_circuit_depth: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimisationConfig {
    pub level: OptLevel,
    pub passes: Vec<String>,
    pub disabled_passes: Vec<String>,
    pub max_iterations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub enable_shape_propagation: bool,
    pub enable_flop_counting: bool,
    pub enable_memory_analysis: bool,
    pub enable_noise_simulation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub topology: String,
    pub num_qubits: usize,
    pub error_mitigation: Option<String>,
    pub shots: Option<usize>,
}

impl Default for LithConfig {
    fn default() -> Self {
        Self {
            target: TargetConfig {
                backend: "llvm".into(),
                device: None,
                precision: Some("fp32".into()),
            },
            budget: BudgetConfig {
                max_flops: None,
                max_memory_bytes: None,
                max_time_ms: None,
                min_fidelity: None,
                max_circuit_depth: None,
            },
            optimisation: OptimisationConfig {
                level: OptLevel::O2,
                passes: vec![
                    "canonicalize".into(),
                    "constant-folding".into(),
                    "dce".into(),
                    "tensor-fusion".into(),
                ],
                disabled_passes: Vec::new(),
                max_iterations: 10,
            },
            simulation: SimulationConfig {
                enable_shape_propagation: true,
                enable_flop_counting: true,
                enable_memory_analysis: true,
                enable_noise_simulation: true,
            },
            quantum: None,
        }
    }
}

impl LithConfig {
    pub fn with_quantum(mut self, topology: &str, num_qubits: usize) -> Self {
        self.quantum = Some(QuantumConfig {
            topology: topology.into(),
            num_qubits,
            error_mitigation: None,
            shots: Some(1024),
        });
        self
    }
}
