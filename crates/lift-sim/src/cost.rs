use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub flops_per_second: f64,
    pub memory_bandwidth_gb_s: f64,
    pub gpu_memory_gb: f64,
    pub num_gpus: usize,
    pub interconnect_bw_gb_s: f64,
}

impl CostModel {
    pub fn a100() -> Self {
        Self {
            flops_per_second: 312e12,  // 312 TFLOPS FP16
            memory_bandwidth_gb_s: 2039.0,
            gpu_memory_gb: 80.0,
            num_gpus: 1,
            interconnect_bw_gb_s: 600.0,
        }
    }

    pub fn h100() -> Self {
        Self {
            flops_per_second: 989e12,
            memory_bandwidth_gb_s: 3350.0,
            gpu_memory_gb: 80.0,
            num_gpus: 1,
            interconnect_bw_gb_s: 900.0,
        }
    }

    pub fn compute_time_ms(&self, flops: u64) -> f64 {
        (flops as f64 / self.flops_per_second) * 1000.0
    }

    pub fn memory_time_ms(&self, bytes: u64) -> f64 {
        let gb = bytes as f64 / 1e9;
        (gb / self.memory_bandwidth_gb_s) * 1000.0
    }

    pub fn roofline_time_ms(&self, flops: u64, bytes: u64) -> f64 {
        let compute_ms = self.compute_time_ms(flops);
        let memory_ms = self.memory_time_ms(bytes);
        compute_ms.max(memory_ms)
    }

    pub fn arithmetic_intensity(&self, flops: u64, bytes: u64) -> f64 {
        if bytes == 0 { return f64::INFINITY; }
        flops as f64 / bytes as f64
    }

    pub fn is_compute_bound(&self, flops: u64, bytes: u64) -> bool {
        let ai = self.arithmetic_intensity(flops, bytes);
        let ridge_point = self.flops_per_second / (self.memory_bandwidth_gb_s * 1e9);
        ai >= ridge_point
    }

    pub fn fits_in_memory(&self, bytes: u64) -> bool {
        (bytes as f64) <= self.gpu_memory_gb * 1e9
    }

    pub fn num_gpus_needed(&self, bytes: u64) -> usize {
        let mem_per_gpu = (self.gpu_memory_gb * 1e9) as u64;
        ((bytes + mem_per_gpu - 1) / mem_per_gpu) as usize
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCostModel {
    pub gate_time_1q_us: f64,
    pub gate_time_2q_us: f64,
    pub measurement_time_us: f64,
    pub fidelity_1q: f64,
    pub fidelity_2q: f64,
    pub t1_us: f64,
    pub t2_us: f64,
    pub num_qubits: usize,
}

impl QuantumCostModel {
    pub fn superconducting_default() -> Self {
        Self {
            gate_time_1q_us: 0.02,
            gate_time_2q_us: 0.3,
            measurement_time_us: 1.0,
            fidelity_1q: 0.999,
            fidelity_2q: 0.99,
            t1_us: 100.0,
            t2_us: 80.0,
            num_qubits: 127,
        }
    }

    pub fn circuit_fidelity(&self, num_1q: usize, num_2q: usize) -> f64 {
        self.fidelity_1q.powi(num_1q as i32) * self.fidelity_2q.powi(num_2q as i32)
    }

    pub fn circuit_time_us(&self, num_1q: usize, num_2q: usize, num_meas: usize, depth: usize) -> f64 {
        let _ = depth;
        num_1q as f64 * self.gate_time_1q_us
            + num_2q as f64 * self.gate_time_2q_us
            + num_meas as f64 * self.measurement_time_us
    }

    pub fn decoherence_fidelity(&self, circuit_time_us: f64) -> f64 {
        let p1 = (-circuit_time_us / self.t1_us).exp();
        let p2 = (-circuit_time_us / self.t2_us).exp();
        (1.0 + p1 + 2.0 * p2) / 4.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    pub max_flops: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_time_ms: Option<f64>,
    pub min_fidelity: Option<f64>,
    pub max_circuit_depth: Option<usize>,
}

impl Budget {
    pub fn check_flops(&self, flops: u64) -> Result<(), String> {
        if let Some(max) = self.max_flops {
            if flops > max {
                return Err(format!("FLOP budget exceeded: {} > {}", flops, max));
            }
        }
        Ok(())
    }

    pub fn check_memory(&self, bytes: u64) -> Result<(), String> {
        if let Some(max) = self.max_memory_bytes {
            if bytes > max {
                return Err(format!("Memory budget exceeded: {} > {} bytes", bytes, max));
            }
        }
        Ok(())
    }

    pub fn check_fidelity(&self, fidelity: f64) -> Result<(), String> {
        if let Some(min) = self.min_fidelity {
            if fidelity < min {
                return Err(format!("Fidelity below threshold: {:.4} < {:.4}", fidelity, min));
            }
        }
        Ok(())
    }
}

/// Energy and carbon estimation model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyModel {
    pub gpu_tdp_watts: f64,
    pub cpu_tdp_watts: f64,
    pub memory_watts: f64,
    pub cooling_pue: f64,
    pub carbon_intensity_g_per_kwh: f64,
}

impl EnergyModel {
    /// A100 GPU energy profile.
    pub fn a100() -> Self {
        Self {
            gpu_tdp_watts: 400.0,
            cpu_tdp_watts: 250.0,
            memory_watts: 50.0,
            cooling_pue: 1.1,
            carbon_intensity_g_per_kwh: 400.0, // world average
        }
    }

    /// H100 GPU energy profile.
    pub fn h100() -> Self {
        Self {
            gpu_tdp_watts: 700.0,
            cpu_tdp_watts: 350.0,
            memory_watts: 60.0,
            cooling_pue: 1.1,
            carbon_intensity_g_per_kwh: 400.0,
        }
    }

    /// Estimate energy in joules for a given execution time.
    pub fn energy_joules(&self, time_ms: f64, num_gpus: usize) -> f64 {
        let total_watts = (self.gpu_tdp_watts * num_gpus as f64)
            + self.cpu_tdp_watts
            + self.memory_watts;
        let with_cooling = total_watts * self.cooling_pue;
        with_cooling * (time_ms / 1000.0)
    }

    /// Estimate energy in kWh.
    pub fn energy_kwh(&self, time_ms: f64, num_gpus: usize) -> f64 {
        self.energy_joules(time_ms, num_gpus) / 3_600_000.0
    }

    /// Estimate CO₂ emissions in grams.
    pub fn carbon_grams(&self, time_ms: f64, num_gpus: usize) -> f64 {
        self.energy_kwh(time_ms, num_gpus) * self.carbon_intensity_g_per_kwh
    }

    /// Quantum system energy (cryogenic cooling dominated).
    pub fn quantum_energy_joules(&self, circuit_time_us: f64, num_qubits: usize) -> f64 {
        // Dilution refrigerator: ~25kW for superconducting systems
        let cryo_watts = 25_000.0;
        let control_watts = 10.0 * num_qubits as f64; // ~10W per qubit control
        let total_watts = cryo_watts + control_watts;
        total_watts * (circuit_time_us / 1_000_000.0)
    }
}

/// Reactive budget: adjusts constraints based on runtime observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactiveBudget {
    pub budget: Budget,
    pub elapsed_ms: f64,
    pub consumed_flops: u64,
    pub consumed_memory: u64,
    pub current_fidelity: f64,
}

impl ReactiveBudget {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            elapsed_ms: 0.0,
            consumed_flops: 0,
            consumed_memory: 0,
            current_fidelity: 1.0,
        }
    }

    /// Record consumption and return remaining budget.
    pub fn consume(&mut self, flops: u64, memory: u64, time_ms: f64, fidelity_factor: f64) {
        self.consumed_flops += flops;
        self.consumed_memory = self.consumed_memory.max(memory); // peak memory
        self.elapsed_ms += time_ms;
        self.current_fidelity *= fidelity_factor;
    }

    /// Check all remaining budget constraints.
    pub fn check_remaining(&self) -> Result<(), String> {
        if let Some(max) = self.budget.max_flops {
            if self.consumed_flops > max {
                return Err(format!("FLOP budget exhausted: {} / {}", self.consumed_flops, max));
            }
        }
        if let Some(max) = self.budget.max_memory_bytes {
            if self.consumed_memory > max {
                return Err(format!("Memory budget exhausted: {} / {} bytes",
                    self.consumed_memory, max));
            }
        }
        if let Some(max) = self.budget.max_time_ms {
            if self.elapsed_ms > max {
                return Err(format!("Time budget exhausted: {:.2} / {:.2} ms",
                    self.elapsed_ms, max));
            }
        }
        if let Some(min) = self.budget.min_fidelity {
            if self.current_fidelity < min {
                return Err(format!("Fidelity below threshold: {:.6} < {:.6}",
                    self.current_fidelity, min));
            }
        }
        Ok(())
    }

    /// Remaining FLOP budget (None = unlimited).
    pub fn remaining_flops(&self) -> Option<u64> {
        self.budget.max_flops.map(|max| max.saturating_sub(self.consumed_flops))
    }

    /// Remaining time budget in ms (None = unlimited).
    pub fn remaining_time_ms(&self) -> Option<f64> {
        self.budget.max_time_ms.map(|max| (max - self.elapsed_ms).max(0.0))
    }

    /// Utilisation ratio (0.0 to 1.0+) for each resource.
    pub fn utilisation(&self) -> BudgetUtilisation {
        BudgetUtilisation {
            flop_ratio: self.budget.max_flops
                .map(|max| self.consumed_flops as f64 / max as f64),
            memory_ratio: self.budget.max_memory_bytes
                .map(|max| self.consumed_memory as f64 / max as f64),
            time_ratio: self.budget.max_time_ms
                .map(|max| self.elapsed_ms / max),
        }
    }
}

/// Budget utilisation report.
#[derive(Debug, Clone)]
pub struct BudgetUtilisation {
    pub flop_ratio: Option<f64>,
    pub memory_ratio: Option<f64>,
    pub time_ratio: Option<f64>,
}

impl QuantumCostModel {
    pub fn trapped_ion_default() -> Self {
        Self {
            gate_time_1q_us: 10.0,
            gate_time_2q_us: 200.0,
            measurement_time_us: 100.0,
            fidelity_1q: 0.9999,
            fidelity_2q: 0.999,
            t1_us: 1_000_000.0, // ~1 second
            t2_us: 500_000.0,
            num_qubits: 32,
        }
    }

    pub fn neutral_atom_default() -> Self {
        Self {
            gate_time_1q_us: 0.5,
            gate_time_2q_us: 1.0,
            measurement_time_us: 5.0,
            fidelity_1q: 0.999,
            fidelity_2q: 0.995,
            t1_us: 5_000.0,
            t2_us: 2_000.0,
            num_qubits: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a100_roofline() {
        let model = CostModel::a100();
        let flops = 2 * 1024 * 1024 * 1024_u64;
        let bytes = 4 * 1024 * 1024_u64;
        let time = model.roofline_time_ms(flops, bytes);
        assert!(time > 0.0);
    }

    #[test]
    fn test_quantum_fidelity() {
        let model = QuantumCostModel::superconducting_default();
        let fidelity = model.circuit_fidelity(10, 5);
        assert!(fidelity > 0.0 && fidelity <= 1.0);
    }

    #[test]
    fn test_budget_check() {
        let budget = Budget {
            max_flops: Some(1_000_000),
            max_memory_bytes: None,
            max_time_ms: None,
            min_fidelity: Some(0.9),
            max_circuit_depth: None,
        };
        assert!(budget.check_flops(500_000).is_ok());
        assert!(budget.check_flops(2_000_000).is_err());
        assert!(budget.check_fidelity(0.95).is_ok());
        assert!(budget.check_fidelity(0.85).is_err());
    }
}
