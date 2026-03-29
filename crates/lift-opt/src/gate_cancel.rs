use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use lift_quantum::gates::QuantumGate;
use std::collections::HashSet;

#[derive(Debug)]
pub struct GateCancellation;

impl Pass for GateCancellation {
    fn name(&self) -> &str { "gate-cancellation" }

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

            for i in 0..op_list.len().saturating_sub(1) {
                let op1_key = op_list[i];
                let op2_key = op_list[i + 1];

                if ops_to_remove.contains(&op1_key) || ops_to_remove.contains(&op2_key) {
                    continue;
                }

                let (gate1, gate2, same_qubit) = {
                    let op1 = match ctx.ops.get(op1_key) { Some(o) => o, None => continue };
                    let op2 = match ctx.ops.get(op2_key) { Some(o) => o, None => continue };

                    let name1 = ctx.strings.resolve(op1.name).to_string();
                    let name2 = ctx.strings.resolve(op2.name).to_string();

                    let g1 = match QuantumGate::from_name(&name1) { Some(g) => g, None => continue };
                    let g2 = match QuantumGate::from_name(&name2) { Some(g) => g, None => continue };

                    // Check if op2's input is op1's output (SSA chain)
                    let same = if !op1.results.is_empty() && !op2.inputs.is_empty() {
                        op1.results.iter().any(|r| op2.inputs.contains(r))
                    } else {
                        false
                    };

                    (g1, g2, same)
                };

                // Cancel self-inverse gates: H·H = I, X·X = I, etc.
                if gate1 == gate2 && gate1.is_self_inverse() && same_qubit {
                    // Rewire: op2's output users should use op1's input
                    let op1 = &ctx.ops[op1_key];
                    let op2 = &ctx.ops[op2_key];

                    if !op1.inputs.is_empty() && !op2.results.is_empty() {
                        let original_input = op1.inputs[0];
                        let final_output = op2.results[0];

                        // Update all uses of final_output to use original_input
                        let op_keys_all: Vec<_> = ctx.ops.keys().collect();
                        for ok in op_keys_all {
                            if ok == op1_key || ok == op2_key { continue; }
                            if let Some(op) = ctx.ops.get_mut(ok) {
                                for input in &mut op.inputs {
                                    if *input == final_output {
                                        *input = original_input;
                                    }
                                }
                            }
                        }

                        ops_to_remove.insert(op1_key);
                        ops_to_remove.insert(op2_key);
                        cancelled += 1;
                    }
                }

                // Cancel S·Sdg = I and T·Tdg = I
                let is_adjoint_pair = matches!(
                    (&gate1, &gate2),
                    (QuantumGate::S, QuantumGate::Sdg) | (QuantumGate::Sdg, QuantumGate::S) |
                    (QuantumGate::T, QuantumGate::Tdg) | (QuantumGate::Tdg, QuantumGate::T)
                );

                if is_adjoint_pair && same_qubit {
                    let op1 = &ctx.ops[op1_key];
                    let op2 = &ctx.ops[op2_key];

                    if !op1.inputs.is_empty() && !op2.results.is_empty() {
                        let original_input = op1.inputs[0];
                        let final_output = op2.results[0];

                        let op_keys_all: Vec<_> = ctx.ops.keys().collect();
                        for ok in op_keys_all {
                            if ok == op1_key || ok == op2_key { continue; }
                            if let Some(op) = ctx.ops.get_mut(ok) {
                                for input in &mut op.inputs {
                                    if *input == final_output {
                                        *input = original_input;
                                    }
                                }
                            }
                        }

                        ops_to_remove.insert(op1_key);
                        ops_to_remove.insert(op2_key);
                        cancelled += 1;
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
