use lift_core::dialect::Dialect;
use crate::ops::HybridOp;

#[derive(Debug)]
pub struct HybridDialect;

impl Dialect for HybridDialect {
    fn name(&self) -> &str { "hybrid" }

    fn verify_op(&self, op_name: &str, _num_inputs: usize, _num_results: usize) -> Result<(), String> {
        let full_name = if op_name.starts_with("hybrid.") {
            op_name.to_string()
        } else {
            format!("hybrid.{}", op_name)
        };

        match HybridOp::from_name(&full_name) {
            Some(_) => Ok(()),
            None => Err(format!("Unknown hybrid operation: {}", full_name)),
        }
    }
}

pub fn register_hybrid_dialect(registry: &mut lift_core::dialect::DialectRegistry) {
    registry.register(Box::new(HybridDialect));
}
