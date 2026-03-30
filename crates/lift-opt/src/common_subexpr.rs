use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use std::collections::HashMap;

/// Common Subexpression Elimination (CSE): detects identical operations
/// with the same inputs and attributes, replacing duplicates with
/// references to the first occurrence.
#[derive(Debug)]
pub struct CommonSubexprElimination;

/// A fingerprint for an operation used to detect duplicates.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct OpFingerprint {
    name: String,
    inputs: Vec<lift_core::values::ValueKey>,
    attrs_hash: String,
}

impl Pass for CommonSubexprElimination {
    fn name(&self) -> &str { "common-subexpr-elimination" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut eliminated = 0usize;
        let mut ops_to_remove = Vec::new();

        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let op_list = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            // Map from fingerprint -> first op's result values
            let mut seen: HashMap<OpFingerprint, Vec<lift_core::values::ValueKey>> = HashMap::new();

            for &op_key in &op_list {
                let fingerprint = match ctx.ops.get(op_key) {
                    Some(op) => {
                        // Skip ops with side effects (measurement, store, etc.)
                        let name_str = ctx.strings.resolve(op.name).to_string();
                        if name_str.contains("measure") || name_str.contains("store") ||
                           name_str.contains("send") || name_str.contains("receive") ||
                           name_str.contains("barrier") || name_str.contains("reset") {
                            continue;
                        }

                        // Skip ops with no results (nothing to deduplicate)
                        if op.results.is_empty() { continue; }

                        let attrs_str = format!("{:?}", op.attrs);
                        OpFingerprint {
                            name: name_str,
                            inputs: op.inputs.clone(),
                            attrs_hash: attrs_str,
                        }
                    }
                    None => continue,
                };

                if let Some(existing_results) = seen.get(&fingerprint) {
                    // Found a duplicate - rewire users
                    let dup_results = match ctx.ops.get(op_key) {
                        Some(op) => op.results.clone(),
                        None => continue,
                    };

                    if dup_results.len() != existing_results.len() { continue; }

                    // Rewire all users of dup_results to use existing_results
                    let all_ops: Vec<_> = ctx.ops.keys().collect();
                    for &ok in &all_ops {
                        if ok == op_key { continue; }
                        if let Some(other) = ctx.ops.get_mut(ok) {
                            for inp in &mut other.inputs {
                                for (idx, dup_r) in dup_results.iter().enumerate() {
                                    if *inp == *dup_r {
                                        *inp = existing_results[idx];
                                    }
                                }
                            }
                        }
                    }

                    ops_to_remove.push(op_key);
                    eliminated += 1;
                } else {
                    let results = match ctx.ops.get(op_key) {
                        Some(op) => op.results.clone(),
                        None => continue,
                    };
                    seen.insert(fingerprint, results);
                }
            }

            // Remove from block
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

        if eliminated > 0 {
            tracing::info!(
                pass = "cse",
                eliminated = eliminated,
                "Common subexpression elimination applied"
            );
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}
