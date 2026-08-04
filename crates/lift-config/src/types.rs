use serde::{Deserialize, Serialize};

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
    pub provider: Option<QuantumProvider>,
}

/// Target hardware provider used for native gate decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumProvider {
    Ibm,
    IbmKyoto,
    Rigetti,
    IonQ,
    Quantinuum,
    Simulator,
}

impl QuantumProvider {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ibm" | "ibm_eagle" | "ibm-eagle" => Some(Self::Ibm),
            "ibm_kyoto" | "ibm-kyoto" => Some(Self::IbmKyoto),
            "rigetti" => Some(Self::Rigetti),
            "ionq" => Some(Self::IonQ),
            "quantinuum" => Some(Self::Quantinuum),
            "simulator" | "sim" => Some(Self::Simulator),
            _ => None,
        }
    }
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
                passes: Vec::new(),
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

impl OptimisationConfig {
    /// The canonical pass names known to the optimiser.
    pub const ALL_PASSES: &'static [&'static str] = &[
        "canonicalize",
        "constant-folding",
        "dce",
        "cse",
        "tensor-fusion",
        "flash-attention",
        "quantisation-pass",
        "gate-cancellation",
        "rotation-merge",
        "noise-aware-schedule",
        "layout-mapping",
        "gate-decomposition",
        "real-routing",
    ];

    /// Returns the pass pipeline for a given optimisation level.
    ///
    /// * `O0` — no optimisation.
    /// * `O1` — safe structural passes (canonicalise, constant folding, DCE).
    /// * `O2` — O1 + tensor fusion and CSE.
    /// * `O3` — O2 + quantum passes (gate cancellation, rotation merge) and
    ///   hardware-oriented passes (noise-aware schedule, layout mapping).
    pub fn passes_for_level(level: OptLevel) -> Vec<String> {
        match level {
            OptLevel::O0 => vec![],
            OptLevel::O1 => vec![
                "canonicalize".to_string(),
                "constant-folding".to_string(),
                "dce".to_string(),
            ],
            OptLevel::O2 => vec![
                "canonicalize".to_string(),
                "constant-folding".to_string(),
                "dce".to_string(),
                "cse".to_string(),
                "tensor-fusion".to_string(),
            ],
            OptLevel::O3 => vec![
                "canonicalize".to_string(),
                "constant-folding".to_string(),
                "dce".to_string(),
                "cse".to_string(),
                "tensor-fusion".to_string(),
                "flash-attention".to_string(),
                "quantisation-pass".to_string(),
                "gate-cancellation".to_string(),
                "rotation-merge".to_string(),
                "noise-aware-schedule".to_string(),
                "layout-mapping".to_string(),
                "gate-decomposition".to_string(),
                "real-routing".to_string(),
            ],
        }
    }

    /// The effective pass list: explicit `passes` take priority; otherwise the
    /// pipeline is derived from `level`. `disabled_passes` are always removed.
    pub fn effective_passes(&self) -> Vec<String> {
        let mut passes = if self.passes.is_empty() {
            Self::passes_for_level(self.level)
        } else {
            self.passes.clone()
        };
        passes.retain(|p| !self.disabled_passes.contains(p));
        passes
    }

    /// Validates that every pass name is known. Returns the unknown names.
    pub fn validate(&self) -> Vec<String> {
        let mut unknown = Vec::new();
        for p in self.effective_passes() {
            if !Self::ALL_PASSES.contains(&p.as_str()) {
                unknown.push(p);
            }
        }
        unknown
    }
}

impl LithConfig {
    pub fn with_quantum(mut self, topology: &str, num_qubits: usize) -> Self {
        self.quantum = Some(QuantumConfig {
            topology: topology.into(),
            num_qubits,
            error_mitigation: None,
            shots: Some(1024),
            provider: None,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passes_for_level_o0_is_empty() {
        assert!(OptimisationConfig::passes_for_level(OptLevel::O0).is_empty());
    }

    #[test]
    fn test_passes_for_level_o3_contains_all() {
        let passes = OptimisationConfig::passes_for_level(OptLevel::O3);
        for p in OptimisationConfig::ALL_PASSES {
            assert!(passes.contains(&p.to_string()), "missing {}", p);
        }
    }

    #[test]
    fn test_effective_passes_uses_level_when_empty() {
        let cfg = OptimisationConfig {
            level: OptLevel::O1,
            passes: vec![],
            disabled_passes: vec![],
            max_iterations: 10,
        };
        assert_eq!(
            cfg.effective_passes(),
            vec![
                "canonicalize".to_string(),
                "constant-folding".to_string(),
                "dce".to_string(),
            ]
        );
    }

    #[test]
    fn test_effective_passes_explicit_overrides_level() {
        let cfg = OptimisationConfig {
            level: OptLevel::O3,
            passes: vec!["dce".to_string(), "cse".to_string()],
            disabled_passes: vec!["cse".to_string()],
            max_iterations: 10,
        };
        assert_eq!(cfg.effective_passes(), vec!["dce".to_string()]);
    }

    #[test]
    fn test_validate_detects_unknown_pass() {
        let cfg = OptimisationConfig {
            level: OptLevel::O1,
            passes: vec!["dce".to_string(), "not-a-pass".to_string()],
            disabled_passes: vec![],
            max_iterations: 10,
        };
        assert_eq!(cfg.validate(), vec!["not-a-pass".to_string()]);
    }
}
