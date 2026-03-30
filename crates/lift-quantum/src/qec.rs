use serde::{Serialize, Deserialize};

/// Quantum Error Correction code types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QecCode {
    SurfaceCode { distance: u32 },
    SteaneCode,
    ShorCode,
    RepetitionCode { distance: u32 },
    LdpcCode { n: u32, k: u32 },
}

/// Result of QEC analysis on a circuit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QecAnalysis {
    pub code: QecCode,
    pub logical_error_rate: f64,
    pub physical_error_rate: f64,
    pub overhead_qubits: u32,
    pub syndrome_depth: u32,
    pub logical_qubits: u32,
    pub physical_qubits: u32,
}

impl QecCode {
    /// Number of physical qubits per logical qubit for this code.
    pub fn physical_per_logical(&self) -> u32 {
        match self {
            Self::SurfaceCode { distance } => distance * distance,
            Self::SteaneCode => 7,
            Self::ShorCode => 9,
            Self::RepetitionCode { distance } => *distance,
            Self::LdpcCode { n, k } => {
                if *k == 0 { return *n; }
                n / k
            }
        }
    }

    /// Code distance (minimum weight of a logical operator).
    pub fn code_distance(&self) -> u32 {
        match self {
            Self::SurfaceCode { distance } => *distance,
            Self::SteaneCode => 3,
            Self::ShorCode => 3,
            Self::RepetitionCode { distance } => *distance,
            Self::LdpcCode { n, .. } => {
                // Approximate: sqrt(n) for good LDPC codes
                (*n as f64).sqrt().ceil() as u32
            }
        }
    }

    /// Syndrome extraction circuit depth per round.
    pub fn syndrome_circuit_depth(&self) -> u32 {
        match self {
            Self::SurfaceCode { distance } => *distance,
            Self::SteaneCode => 4,
            Self::ShorCode => 8,
            Self::RepetitionCode { .. } => 2,
            Self::LdpcCode { .. } => 6,
        }
    }
}

impl QecAnalysis {
    /// Analyse a circuit's QEC requirements.
    ///
    /// # Arguments
    /// * `num_logical_qubits` - Number of logical qubits in the circuit
    /// * `circuit_depth` - Circuit depth (number of time steps)
    /// * `code` - The QEC code to use
    /// * `phys_error_rate` - Physical gate error rate (e.g. 0.001)
    pub fn analyse(
        num_logical_qubits: u32,
        circuit_depth: u32,
        code: QecCode,
        phys_error_rate: f64,
    ) -> Self {
        let d = code.code_distance();
        let physical_per_logical = code.physical_per_logical();
        let physical_qubits = num_logical_qubits * physical_per_logical;

        // Logical error rate model:
        // p_L ≈ A * (p / p_th)^((d+1)/2)
        // where p_th ≈ 0.01 for surface codes
        let p_threshold = match &code {
            QecCode::SurfaceCode { .. } => 0.01,
            QecCode::SteaneCode => 0.005,
            QecCode::ShorCode => 0.003,
            QecCode::RepetitionCode { .. } => 0.03,
            QecCode::LdpcCode { .. } => 0.008,
        };

        let ratio = phys_error_rate / p_threshold;
        let exponent = (d as f64 + 1.0) / 2.0;
        let a_prefactor = 0.1; // empirical prefactor
        let logical_error_per_round = a_prefactor * ratio.powf(exponent);

        // Total logical error rate over the circuit
        let total_rounds = circuit_depth;
        let logical_error_rate =
            1.0 - (1.0 - logical_error_per_round).powi(total_rounds as i32);

        let syndrome_depth = code.syndrome_circuit_depth() * total_rounds;

        Self {
            code,
            logical_error_rate,
            physical_error_rate: phys_error_rate,
            overhead_qubits: physical_per_logical,
            syndrome_depth,
            logical_qubits: num_logical_qubits,
            physical_qubits,
        }
    }

    /// Returns `true` if the logical error rate is below the given threshold.
    pub fn meets_target(&self, target_error_rate: f64) -> bool {
        self.logical_error_rate <= target_error_rate
    }

    /// Suggest the minimum code distance needed to achieve a target logical error rate.
    pub fn suggest_distance(
        phys_error_rate: f64,
        target_logical_rate: f64,
        circuit_depth: u32,
    ) -> u32 {
        let p_threshold = 0.01; // surface code threshold
        let ratio = phys_error_rate / p_threshold;

        if ratio >= 1.0 {
            // Below threshold — QEC cannot help
            return 0;
        }

        // Binary search for minimum d
        for d in (3..=49).step_by(2) {
            let exponent = (d as f64 + 1.0) / 2.0;
            let per_round = 0.1 * ratio.powf(exponent);
            let total = 1.0 - (1.0 - per_round).powi(circuit_depth as i32);
            if total <= target_logical_rate {
                return d;
            }
        }
        51 // Very large distance needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_code_overhead() {
        let code = QecCode::SurfaceCode { distance: 5 };
        assert_eq!(code.physical_per_logical(), 25);
        assert_eq!(code.code_distance(), 5);
    }

    #[test]
    fn test_steane_code() {
        let code = QecCode::SteaneCode;
        assert_eq!(code.physical_per_logical(), 7);
        assert_eq!(code.code_distance(), 3);
    }

    #[test]
    fn test_shor_code() {
        let code = QecCode::ShorCode;
        assert_eq!(code.physical_per_logical(), 9);
        assert_eq!(code.code_distance(), 3);
    }

    #[test]
    fn test_qec_analysis_below_threshold() {
        let analysis = QecAnalysis::analyse(
            10,   // 10 logical qubits
            100,  // 100 depth
            QecCode::SurfaceCode { distance: 7 },
            0.001, // 0.1% physical error
        );
        assert!(analysis.logical_error_rate < 1.0);
        assert!(analysis.logical_error_rate >= 0.0);
        assert_eq!(analysis.physical_qubits, 10 * 49);
    }

    #[test]
    fn test_qec_meets_target() {
        let analysis = QecAnalysis::analyse(
            5, 50,
            QecCode::SurfaceCode { distance: 11 },
            0.001,
        );
        // With d=11 and p=0.001, should be very low error
        assert!(analysis.meets_target(0.01));
    }

    #[test]
    fn test_suggest_distance() {
        let d = QecAnalysis::suggest_distance(0.001, 1e-6, 100);
        assert!(d >= 3);
        assert!(d <= 49);
    }

    #[test]
    fn test_repetition_code() {
        let code = QecCode::RepetitionCode { distance: 5 };
        assert_eq!(code.physical_per_logical(), 5);
        assert_eq!(code.code_distance(), 5);
    }
}
