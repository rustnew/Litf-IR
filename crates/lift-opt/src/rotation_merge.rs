use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use lift_quantum::gates::QuantumGate;
use std::collections::HashSet;

/// Rotation merge pass: Rz(a)·Rz(b) → Rz(a+b), and remove identity rotations.
/// Also applies to Rx and Ry.
#[derive(Debug)]
pub struct RotationMerge;

impl Pass for RotationMerge {
    fn name(&self) -> &str { "rotation-merge" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut merged = 0usize;
        let mut removed = 0usize;
        let mut ops_to_remove: HashSet<lift_core::operations::OpKey> = HashSet::new();

        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let block = match ctx.blocks.get(block_key) {
                Some(b) => b,
                None => continue,
            };
            let op_list: Vec<_> = block.ops.clone();

            // Pass 1: Remove identity rotations (angle ≈ 0 or ≈ 2π)
            for &op_key in &op_list {
                if ops_to_remove.contains(&op_key) { continue; }

                let is_identity_rotation = {
                    let op = match ctx.ops.get(op_key) { Some(o) => o, None => continue };
                    let name = ctx.strings.resolve(op.name).to_string();
                    let gate = match QuantumGate::from_name(&name) { Some(g) => g, None => continue };

                    if !matches!(gate, QuantumGate::RX | QuantumGate::RY | QuantumGate::RZ) {
                        false
                    } else {
                        let angle = op.attrs.get_float("angle").unwrap_or(0.0);
                        let norm = angle.rem_euclid(std::f64::consts::TAU);
                        norm.abs() < 1e-12 || (norm - std::f64::consts::TAU).abs() < 1e-12
                    }
                };

                if is_identity_rotation {
                    // Rewire: users of the output should use the input directly
                    let op = match ctx.ops.get(op_key) { Some(o) => o, None => continue };
                    if !op.inputs.is_empty() && !op.results.is_empty() {
                        let original_input = op.inputs[0];
                        let output = op.results[0];

                        let all_ops: Vec<_> = ctx.ops.keys().collect();
                        for ok in all_ops {
                            if ok == op_key { continue; }
                            if let Some(other) = ctx.ops.get_mut(ok) {
                                for inp in &mut other.inputs {
                                    if *inp == output { *inp = original_input; }
                                }
                            }
                        }
                        ops_to_remove.insert(op_key);
                        removed += 1;
                    }
                }
            }

            // Pass 2: Merge consecutive same-axis rotations
            let op_list: Vec<_> = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            for i in 0..op_list.len().saturating_sub(1) {
                let op1_key = op_list[i];
                let op2_key = op_list[i + 1];

                if ops_to_remove.contains(&op1_key) || ops_to_remove.contains(&op2_key) {
                    continue;
                }

                let merge_info = {
                    let op1 = match ctx.ops.get(op1_key) { Some(o) => o, None => continue };
                    let op2 = match ctx.ops.get(op2_key) { Some(o) => o, None => continue };

                    let name1 = ctx.strings.resolve(op1.name).to_string();
                    let name2 = ctx.strings.resolve(op2.name).to_string();

                    if name1 != name2 { continue; }

                    let gate = match QuantumGate::from_name(&name1) { Some(g) => g, None => continue };
                    if !matches!(gate, QuantumGate::RX | QuantumGate::RY | QuantumGate::RZ) {
                        continue;
                    }

                    // Check SSA chain: op2 uses op1's result
                    let same_qubit = if !op1.results.is_empty() && !op2.inputs.is_empty() {
                        op1.results.iter().any(|r| op2.inputs.contains(r))
                    } else {
                        false
                    };

                    if !same_qubit { continue; }

                    let a1 = op1.attrs.get_float("angle").unwrap_or(0.0);
                    let a2 = op2.attrs.get_float("angle").unwrap_or(0.0);
                    Some(a1 + a2)
                };

                if let Some(merged_angle) = merge_info {
                    // Update op1 with merged angle
                    if let Some(op1) = ctx.ops.get_mut(op1_key) {
                        op1.attrs.set("angle", lift_core::attributes::Attribute::Float(merged_angle));
                    }

                    // Rewire op2's result users to use op1's result
                    let (op1_result, op2_result) = {
                        let op1 = match ctx.ops.get(op1_key) { Some(o) => o, None => continue };
                        let op2 = match ctx.ops.get(op2_key) { Some(o) => o, None => continue };
                        if op1.results.is_empty() || op2.results.is_empty() { continue; }
                        (op1.results[0], op2.results[0])
                    };

                    let all_ops: Vec<_> = ctx.ops.keys().collect();
                    for ok in all_ops {
                        if ok == op1_key || ok == op2_key { continue; }
                        if let Some(other) = ctx.ops.get_mut(ok) {
                            for inp in &mut other.inputs {
                                if *inp == op2_result { *inp = op1_result; }
                            }
                        }
                    }

                    ops_to_remove.insert(op2_key);
                    merged += 1;
                }
            }

            // Remove ops from block
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

        if merged > 0 || removed > 0 {
            tracing::info!(
                pass = "rotation-merge",
                merged = merged,
                identity_removed = removed,
                "Rotation merge applied"
            );
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis", "quantum_analysis"]
    }
}
