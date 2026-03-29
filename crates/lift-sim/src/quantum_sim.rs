use lift_core::context::Context;
use lift_quantum::noise::{CircuitNoise, GateNoise};
use lift_quantum::gates::QuantumGate;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantumAnalysis {
    pub num_qubits_used: usize,
    pub gate_count: usize,
    pub one_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub three_qubit_gates: usize,
    pub measurements: usize,
    pub circuit_depth: usize,
    pub estimated_fidelity: f64,
    pub noise: CircuitNoise,
}

pub fn analyze_quantum_ops(ctx: &Context) -> QuantumAnalysis {
    let mut analysis = QuantumAnalysis {
        estimated_fidelity: 1.0,
        noise: CircuitNoise::new(),
        ..Default::default()
    };

    for (_op_key, op) in &ctx.ops {
        let op_name = ctx.strings.resolve(op.name).to_string();

        if let Some(gate) = QuantumGate::from_name(&op_name) {
            analysis.gate_count += 1;
            match gate.num_qubits() {
                1 => {
                    analysis.one_qubit_gates += 1;
                    let noise = GateNoise::with_depolarizing(0.999, 0.02);
                    analysis.noise.add_gate(&noise, false);
                }
                2 => {
                    analysis.two_qubit_gates += 1;
                    let noise = GateNoise::with_depolarizing(0.99, 0.3);
                    analysis.noise.add_gate(&noise, true);
                }
                3 => {
                    analysis.three_qubit_gates += 1;
                    let noise = GateNoise::with_depolarizing(0.98, 0.6);
                    analysis.noise.add_gate(&noise, false);
                }
                _ => {}
            }

            if matches!(gate, QuantumGate::Measure | QuantumGate::MeasureAll) {
                analysis.measurements += 1;
            }
        }
    }

    // Count qubits used
    for (_val_key, val) in &ctx.values {
        if ctx.is_qubit_type(val.ty) {
            analysis.num_qubits_used += 1;
        }
    }
    // Each qubit is defined multiple times in SSA form, estimate unique qubits
    // as the number of qubit-typed block args
    let mut qubit_block_args = 0;
    for (_block_key, block) in &ctx.blocks {
        for &arg in &block.args {
            if let Some(val) = ctx.get_value(arg) {
                if ctx.is_qubit_type(val.ty) {
                    qubit_block_args += 1;
                }
            }
        }
    }
    if qubit_block_args > 0 {
        analysis.num_qubits_used = qubit_block_args;
    }

    analysis.estimated_fidelity = analysis.noise.total_fidelity;
    analysis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_quantum_analysis() {
        let ctx = Context::new();
        let analysis = analyze_quantum_ops(&ctx);
        assert_eq!(analysis.gate_count, 0);
        assert_eq!(analysis.estimated_fidelity, 1.0);
    }
}
