use lift_core::dialect::Dialect;
use crate::ops::TensorOp;

#[derive(Debug)]
pub struct TensorDialect;

impl Dialect for TensorDialect {
    fn name(&self) -> &str { "tensor" }

    fn verify_op(&self, op_name: &str, num_inputs: usize, num_results: usize) -> Result<(), String> {
        let full_name = if op_name.starts_with("tensor.") {
            op_name.to_string()
        } else {
            format!("tensor.{}", op_name)
        };

        match TensorOp::from_name(&full_name) {
            Some(op) => {
                let (min_inputs, max_inputs) = op.num_inputs();
                if num_inputs < min_inputs || num_inputs > max_inputs {
                    return Err(format!(
                        "Operation {} expects {}-{} inputs, got {}",
                        full_name, min_inputs, max_inputs, num_inputs
                    ));
                }
                let _ = num_results;
                Ok(())
            }
            None => Err(format!("Unknown tensor operation: {}", full_name)),
        }
    }
}

pub fn register_tensor_dialect(registry: &mut lift_core::dialect::DialectRegistry) {
    registry.register(Box::new(TensorDialect));
}
