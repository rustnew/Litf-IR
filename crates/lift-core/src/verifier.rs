use std::collections::HashSet;
use thiserror::Error;
use crate::context::Context;
use crate::values::ValueKey;
use crate::operations::OpKey;
use crate::blocks::BlockKey;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("SSA violation: value {0:?} used but not defined")]
    UndefinedValue(ValueKey),

    #[error("SSA violation: value {0:?} defined more than once")]
    MultipleDefinition(ValueKey),

    #[error("Dominance violation: value {0:?} used before definition in block {1:?}")]
    DominanceViolation(ValueKey, BlockKey),

    #[error("Type mismatch in operation {op:?}: expected {expected}, got {actual}")]
    TypeMismatch {
        op: OpKey,
        expected: String,
        actual: String,
    },

    #[error("Linearity violation: qubit value {0:?} consumed more than once")]
    LinearityViolation(ValueKey),

    #[error("Linearity violation: qubit {0:?} not consumed (leaked)")]
    QubitLeaked(ValueKey),

    #[error("Branch linearity: arms consume different qubit sets at block {0:?}")]
    BranchLinearityMismatch(BlockKey),

    #[error("Dangling reference: {0}")]
    DanglingReference(String),

    #[error("Empty block {0:?} has no terminator")]
    MissingTerminator(BlockKey),

    #[error("Operation {0:?} has no parent block")]
    OrphanedOperation(OpKey),

    #[error("Block {0:?} has no parent region")]
    OrphanedBlock(BlockKey),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

pub struct Verifier<'a> {
    ctx: &'a Context,
    errors: Vec<VerifyError>,
    defined: HashSet<ValueKey>,
    consumed_qubits: HashSet<ValueKey>,
}

impl<'a> Verifier<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            errors: Vec::new(),
            defined: HashSet::new(),
            consumed_qubits: HashSet::new(),
        }
    }

    pub fn verify_all(&mut self) -> Result<(), Vec<VerifyError>> {
        self.verify_ssa();
        self.verify_well_formedness();
        self.verify_linearity();

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn verify_ssa(&mut self) {
        let mut all_defined: HashSet<ValueKey> = HashSet::new();

        // Collect all defined values from block args
        for (block_key, block) in &self.ctx.blocks {
            for &arg_key in &block.args {
                if !all_defined.insert(arg_key) {
                    self.errors.push(VerifyError::MultipleDefinition(arg_key));
                }
            }
            let _ = block_key;
        }

        // Collect all defined values from operation results
        for (_op_key, op) in &self.ctx.ops {
            for &result_key in &op.results {
                if !all_defined.insert(result_key) {
                    self.errors.push(VerifyError::MultipleDefinition(result_key));
                }
            }
        }

        // Verify all uses are defined
        for (_op_key, op) in &self.ctx.ops {
            for &input_key in &op.inputs {
                if !all_defined.contains(&input_key) {
                    self.errors.push(VerifyError::UndefinedValue(input_key));
                }
            }
        }

        self.defined = all_defined;
    }

    fn verify_well_formedness(&mut self) {
        // Verify all operation inputs reference valid values
        for (op_key, op) in &self.ctx.ops {
            for &input in &op.inputs {
                if !self.ctx.values.contains_key(input) {
                    self.errors.push(VerifyError::DanglingReference(
                        format!("Operation {:?} references non-existent value {:?}", op_key, input),
                    ));
                }
            }
            for &result in &op.results {
                if !self.ctx.values.contains_key(result) {
                    self.errors.push(VerifyError::DanglingReference(
                        format!("Operation {:?} references non-existent result {:?}", op_key, result),
                    ));
                }
            }
            for &region in &op.regions {
                if !self.ctx.regions.contains_key(region) {
                    self.errors.push(VerifyError::DanglingReference(
                        format!("Operation {:?} references non-existent region {:?}", op_key, region),
                    ));
                }
            }
        }

        // Verify blocks reference valid operations
        for (block_key, block) in &self.ctx.blocks {
            for &op in &block.ops {
                if !self.ctx.ops.contains_key(op) {
                    self.errors.push(VerifyError::DanglingReference(
                        format!("Block {:?} references non-existent operation {:?}", block_key, op),
                    ));
                }
            }
        }

        // Verify regions reference valid blocks
        for (region_key, region) in &self.ctx.regions {
            for &block in &region.blocks {
                if !self.ctx.blocks.contains_key(block) {
                    self.errors.push(VerifyError::DanglingReference(
                        format!("Region {:?} references non-existent block {:?}", region_key, block),
                    ));
                }
            }
        }
    }

    fn verify_linearity(&mut self) {
        let mut consumed: HashSet<ValueKey> = HashSet::new();
        let mut all_qubits: HashSet<ValueKey> = HashSet::new();

        // Identify all qubit values
        for (val_key, val) in &self.ctx.values {
            if self.ctx.is_qubit_type(val.ty) {
                all_qubits.insert(val_key);
            }
        }

        // Check each operation's inputs for qubit consumption
        for (_op_key, op) in &self.ctx.ops {
            let op_name = self.ctx.strings.resolve(op.name);

            for &input in &op.inputs {
                if let Some(val) = self.ctx.values.get(input) {
                    if self.ctx.is_qubit_type(val.ty) {
                        if !consumed.insert(input) {
                            self.errors.push(VerifyError::LinearityViolation(input));
                        }
                    }
                }
            }

            // quantum.measure and quantum.reset consume the qubit
            // All gate operations produce new qubit values (SSA)
            if op_name == "quantum.measure" {
                // Qubit is consumed, classical bit produced — no new qubit
            }
        }

        self.consumed_qubits = consumed;
    }

    pub fn errors(&self) -> &[VerifyError] {
        &self.errors
    }
}

pub fn verify(ctx: &Context) -> Result<(), Vec<VerifyError>> {
    let mut verifier = Verifier::new(ctx);
    verifier.verify_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_context_verifies() {
        let ctx = Context::new();
        assert!(verify(&ctx).is_ok());
    }

    #[test]
    fn test_simple_ssa_valid() {
        let mut ctx = Context::new();
        let f32_ty = ctx.make_float_type(32);

        let block = ctx.create_block();
        let arg = ctx.create_block_arg(block, f32_ty);

        let (op, _results) = ctx.create_op(
            "tensor.relu",
            "tensor",
            vec![arg],
            vec![f32_ty],
            crate::attributes::Attributes::new(),
            crate::location::Location::unknown(),
        );
        ctx.add_op_to_block(block, op);

        assert!(verify(&ctx).is_ok());
    }

    #[test]
    fn test_qubit_linearity_violation() {
        let mut ctx = Context::new();
        let qubit_ty = ctx.make_qubit_type();

        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit_ty);

        // First use of q0 — ok
        let (op1, _) = ctx.create_op(
            "quantum.x",
            "quantum",
            vec![q0],
            vec![qubit_ty],
            crate::attributes::Attributes::new(),
            crate::location::Location::unknown(),
        );
        ctx.add_op_to_block(block, op1);

        // Second use of q0 — linearity violation!
        let (op2, _) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![q0],
            vec![qubit_ty],
            crate::attributes::Attributes::new(),
            crate::location::Location::unknown(),
        );
        ctx.add_op_to_block(block, op2);

        let result = verify(&ctx);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(e, VerifyError::LinearityViolation(_))));
    }
}
