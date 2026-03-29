use lift_core::dialect::Dialect;
use crate::gates::QuantumGate;

#[derive(Debug)]
pub struct QuantumDialect;

impl Dialect for QuantumDialect {
    fn name(&self) -> &str { "quantum" }

    fn verify_op(&self, op_name: &str, num_inputs: usize, _num_results: usize) -> Result<(), String> {
        let full_name = if op_name.starts_with("quantum.") {
            op_name.to_string()
        } else {
            format!("quantum.{}", op_name)
        };

        match QuantumGate::from_name(&full_name) {
            Some(gate) => {
                let expected = gate.num_qubits();
                if expected > 0 && num_inputs != expected {
                    return Err(format!(
                        "{} expects {} qubit(s), got {}", full_name, expected, num_inputs
                    ));
                }
                Ok(())
            }
            None => Err(format!("Unknown quantum operation: {}", full_name)),
        }
    }
}

pub fn register_quantum_dialect(registry: &mut lift_core::dialect::DialectRegistry) {
    registry.register(Box::new(QuantumDialect));
}
