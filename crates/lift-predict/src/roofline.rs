use lift_sim::cost::CostModel;
use lift_sim::analysis::AnalysisReport;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineResult {
    pub compute_time_ms: f64,
    pub memory_time_ms: f64,
    pub predicted_time_ms: f64,
    pub arithmetic_intensity: f64,
    pub is_compute_bound: bool,
    pub bottleneck: String,
}

pub fn predict_performance(report: &AnalysisReport, cost_model: &CostModel) -> RooflineResult {
    let compute_ms = cost_model.compute_time_ms(report.total_flops);
    let memory_ms = cost_model.memory_time_ms(report.total_memory_bytes);
    let predicted_ms = compute_ms.max(memory_ms);
    let ai = cost_model.arithmetic_intensity(report.total_flops, report.total_memory_bytes);
    let compute_bound = cost_model.is_compute_bound(report.total_flops, report.total_memory_bytes);

    RooflineResult {
        compute_time_ms: compute_ms,
        memory_time_ms: memory_ms,
        predicted_time_ms: predicted_ms,
        arithmetic_intensity: ai,
        is_compute_bound: compute_bound,
        bottleneck: if compute_bound { "compute".into() } else { "memory".into() },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPrediction {
    pub estimated_fidelity: f64,
    pub circuit_time_us: f64,
    pub num_shots_for_precision: usize,
    pub total_execution_time_ms: f64,
}

pub fn predict_quantum(
    analysis: &lift_sim::quantum_sim::QuantumAnalysis,
    cost_model: &lift_sim::cost::QuantumCostModel,
    target_precision: f64,
) -> QuantumPrediction {
    let circuit_time = cost_model.circuit_time_us(
        analysis.one_qubit_gates,
        analysis.two_qubit_gates,
        analysis.measurements,
        analysis.circuit_depth,
    );

    let decoherence_fid = cost_model.decoherence_fidelity(circuit_time);
    let gate_fid = analysis.estimated_fidelity;
    let total_fidelity = gate_fid * decoherence_fid;

    // Shots needed: 1/precision^2 / fidelity
    let num_shots = ((1.0 / (target_precision * target_precision)) / total_fidelity).ceil() as usize;
    let total_time_ms = (num_shots as f64 * circuit_time) / 1000.0;

    QuantumPrediction {
        estimated_fidelity: total_fidelity,
        circuit_time_us: circuit_time,
        num_shots_for_precision: num_shots,
        total_execution_time_ms: total_time_ms,
    }
}
