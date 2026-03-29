use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use std::collections::HashSet;

#[derive(Debug)]
pub struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &str { "dce" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut used_values: HashSet<lift_core::values::ValueKey> = HashSet::new();

        // Mark phase: find all values used as inputs
        for (_op_key, op) in &ctx.ops {
            for &input in &op.inputs {
                used_values.insert(input);
            }
        }

        // Sweep phase: find ops whose results are never used
        let mut dead_ops: Vec<lift_core::operations::OpKey> = Vec::new();
        for (op_key, op) in &ctx.ops {
            let op_name = ctx.strings.resolve(op.name);
            // Never remove terminators or side-effecting ops
            if op_name == "core.return" || op_name == "core.br" || op_name == "core.cond_br" {
                continue;
            }
            // Don't remove quantum ops (they have side effects on qubits)
            if op_name.starts_with("quantum.") {
                continue;
            }

            if !op.results.is_empty() && op.results.iter().all(|r| !used_values.contains(r)) {
                dead_ops.push(op_key);
            }
        }

        if dead_ops.is_empty() {
            return PassResult::Unchanged;
        }

        // Remove dead ops from their parent blocks
        let dead_set: HashSet<_> = dead_ops.iter().copied().collect();
        for (_block_key, block) in &mut ctx.blocks {
            block.ops.retain(|op| !dead_set.contains(op));
        }

        // Remove the ops and their result values
        for op_key in &dead_ops {
            if let Some(op) = ctx.ops.remove(*op_key) {
                for result in &op.results {
                    ctx.values.remove(*result);
                }
            }
        }

        tracing::info!("DCE removed {} dead operations", dead_ops.len());
        PassResult::Changed
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::attributes::Attributes;
    use lift_core::location::Location;

    #[test]
    fn test_dce_removes_unused_ops() {
        let mut ctx = Context::new();
        let f32_ty = ctx.make_float_type(32);

        let block = ctx.create_block();
        let arg = ctx.create_block_arg(block, f32_ty);

        // This op's result is never used -> dead
        let (dead_op, _dead_results) = ctx.create_op(
            "tensor.relu", "tensor",
            vec![arg], vec![f32_ty],
            Attributes::new(), Location::unknown(),
        );
        ctx.add_op_to_block(block, dead_op);

        // Return the original arg
        let (ret_op, _) = ctx.create_op(
            "core.return", "core",
            vec![arg], vec![],
            Attributes::new(), Location::unknown(),
        );
        ctx.add_op_to_block(block, ret_op);

        let mut cache = AnalysisCache::new();
        let result = DeadCodeElimination.run(&mut ctx, &mut cache);
        assert_eq!(result, PassResult::Changed);

        // Only the return should remain
        let block_data = ctx.get_block(block).unwrap();
        assert_eq!(block_data.ops.len(), 1);
    }
}
