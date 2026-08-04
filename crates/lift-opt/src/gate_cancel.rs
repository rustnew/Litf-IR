use lift_core::context::Context;
use lift_core::pass::{AnalysisCache, Pass, PassResult};
use lift_quantum::gates::QuantumGate;
use std::collections::HashSet;

#[derive(Debug)]
pub struct GateCancellation;

impl Pass for GateCancellation {
    fn name(&self) -> &str {
        "gate-cancellation"
    }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut cancelled = 0usize;
        let mut ops_to_remove: HashSet<lift_core::operations::OpKey> = HashSet::new();

        // For each block, look for consecutive self-inverse gates on the same qubit
        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let block = match ctx.blocks.get(block_key) {
                Some(b) => b,
                None => continue,
            };

            let op_list: Vec<_> = block.ops.clone();

            for i in 0..op_list.len() {
                let op1_key = op_list[i];

                if ops_to_remove.contains(&op1_key) {
                    continue;
                }

                // Inspect every later op (non-consecutive pairs commute when
                // they act on the same SSA chain and the gates in between act
                // on other qubits).
                for &op2_key in op_list.iter().skip(i + 1) {
                    if ops_to_remove.contains(&op2_key) {
                        continue;
                    }

                    let (gate1, gate2, same_qubit) = {
                        let op1 = match ctx.ops.get(op1_key) {
                            Some(o) => o,
                            None => continue,
                        };
                        let op2 = match ctx.ops.get(op2_key) {
                            Some(o) => o,
                            None => continue,
                        };

                        let name1 = ctx.strings.resolve(op1.name).to_string();
                        let name2 = ctx.strings.resolve(op2.name).to_string();

                        let g1 = match QuantumGate::from_name(&name1) {
                            Some(g) => g,
                            None => continue,
                        };
                        let g2 = match QuantumGate::from_name(&name2) {
                            Some(g) => g,
                            None => continue,
                        };

                        // Check if op2's input is op1's output (SSA chain)
                        let same = if !op1.results.is_empty() && !op2.inputs.is_empty() {
                            op1.results.iter().any(|r| op2.inputs.contains(r))
                        } else {
                            false
                        };

                        (g1, g2, same)
                    };

                    // Cancel self-inverse gates: H·H = I, X·X = I, etc.
                    if gate1 == gate2
                        && gate1.is_self_inverse()
                        && same_qubit
                        && cancel_pair(ctx, op1_key, op2_key, &mut ops_to_remove)
                    {
                        cancelled += 1;
                        break;
                    }

                    // Cancel S·Sdg = I and T·Tdg = I
                    let is_adjoint_pair = matches!(
                        (&gate1, &gate2),
                        (QuantumGate::S, QuantumGate::Sdg)
                            | (QuantumGate::Sdg, QuantumGate::S)
                            | (QuantumGate::T, QuantumGate::Tdg)
                            | (QuantumGate::Tdg, QuantumGate::T)
                    );

                    if is_adjoint_pair
                        && same_qubit
                        && cancel_pair(ctx, op1_key, op2_key, &mut ops_to_remove)
                    {
                        cancelled += 1;
                        break;
                    }
                }
            }

            // Remove cancelled ops from the block
            if !ops_to_remove.is_empty() {
                if let Some(block) = ctx.blocks.get_mut(block_key) {
                    block.ops.retain(|op| !ops_to_remove.contains(op));
                }
            }
        }

        // Remove from slotmap
        for op_key in &ops_to_remove {
            if let Some(op) = ctx.ops.remove(*op_key) {
                for result in &op.results {
                    ctx.values.remove(*result);
                }
            }
        }

        if cancelled > 0 {
            tracing::info!("Gate cancellation: cancelled {} gate pairs", cancelled);
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis", "quantum_analysis"]
    }
}

/// Rewires the SSA chain so users of `op2`'s result use `op1`'s input, then
/// marks both ops for removal. Returns `true` if a cancellation happened.
fn cancel_pair(
    ctx: &mut Context,
    op1_key: lift_core::operations::OpKey,
    op2_key: lift_core::operations::OpKey,
    ops_to_remove: &mut HashSet<lift_core::operations::OpKey>,
) -> bool {
    let (op1_input, op2_result) = {
        let op1 = match ctx.ops.get(op1_key) {
            Some(o) => o,
            None => return false,
        };
        let op2 = match ctx.ops.get(op2_key) {
            Some(o) => o,
            None => return false,
        };
        if op1.inputs.is_empty() || op2.results.is_empty() {
            return false;
        }
        (op1.inputs[0], op2.results[0])
    };

    // Update all uses of op2's result to use op1's input.
    let op_keys_all: Vec<_> = ctx.ops.keys().collect();
    for ok in op_keys_all {
        if ok == op1_key || ok == op2_key {
            continue;
        }
        if let Some(op) = ctx.ops.get_mut(ok) {
            for input in &mut op.inputs {
                if *input == op2_result {
                    *input = op1_input;
                }
            }
        }
    }

    ops_to_remove.insert(op1_key);
    ops_to_remove.insert(op2_key);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::attributes::Attributes;
    use lift_core::location::Location;

    /// Builds: H(q0) -> q1, X(q2) -> q3 (other qubit), H(q1) -> q4
    /// The two H gates are non-consecutive but on the same SSA chain, so they
    /// should cancel (the X in between acts on a different qubit and commutes).
    #[test]
    fn test_non_consecutive_cancellation() {
        let mut ctx = Context::new();
        let qubit = ctx.make_qubit_type();
        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit);
        let q2 = ctx.create_block_arg(block, qubit);

        let (h1, h1_res) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![q0],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h1);

        // Interleaved gate on another qubit.
        let (x, _) = ctx.create_op(
            "quantum.x",
            "quantum",
            vec![q2],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, x);

        let (h2, h2_res) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![h1_res[0]],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h2);

        // Consumer of the final H output.
        let (cx, _) = ctx.create_op(
            "quantum.cx",
            "quantum",
            vec![h2_res[0], q2],
            vec![qubit, qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, cx);

        let result = GateCancellation.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        // Both H ops should be gone.
        let h_count = ctx
            .ops
            .values()
            .filter(|op| ctx.strings.resolve(op.name) == "quantum.h")
            .count();
        assert_eq!(h_count, 0);

        // The CX should now consume q0 directly (rewired).
        let cx_op = ctx
            .ops
            .values()
            .find(|op| ctx.strings.resolve(op.name) == "quantum.cx")
            .unwrap();
        assert!(cx_op.inputs.contains(&q0));
    }

    /// H(q0) -> q1, X(q1) -> q2 (same qubit!), H(q2) -> q3 must NOT cancel
    /// because H·X·H != I.
    #[test]
    fn test_no_cancel_when_intermediate_same_qubit() {
        let mut ctx = Context::new();
        let qubit = ctx.make_qubit_type();
        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit);

        let (h1, h1_res) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![q0],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h1);

        let (x, x_res) = ctx.create_op(
            "quantum.x",
            "quantum",
            vec![h1_res[0]],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, x);

        let (h2, _) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![x_res[0]],
            vec![qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h2);

        let result = GateCancellation.run(&mut ctx, &mut AnalysisCache::new());
        assert_eq!(result, PassResult::Unchanged);

        let h_count = ctx
            .ops
            .values()
            .filter(|op| ctx.strings.resolve(op.name) == "quantum.h")
            .count();
        assert_eq!(h_count, 2);
    }
}
