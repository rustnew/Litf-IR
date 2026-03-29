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
