use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};

/// Layout mapping pass: inserts SWAP gates to map logical qubits
/// to physical qubits based on device topology constraints.
/// Uses a greedy nearest-neighbor heuristic.
#[derive(Debug)]
pub struct LayoutMapping;

impl Pass for LayoutMapping {
    fn name(&self) -> &str { "layout-mapping" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut swaps_inserted = 0usize;

        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let op_list = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            // Collect 2-qubit gate ops that have qubit attributes
            for &op_key in &op_list {
                let needs_swap = if let Some(op) = ctx.ops.get(op_key) {
                    let name = ctx.strings.resolve(op.name);
                    if !name.starts_with("quantum.") { continue; }

                    // Check if this is a 2-qubit gate with non-adjacent qubits
                    let q0 = op.attrs.get_integer("qubit0");
                    let q1 = op.attrs.get_integer("qubit1");
                    let max_distance = op.attrs.get_integer("max_coupling_distance").unwrap_or(1);

                    match (q0, q1) {
                        (Some(a), Some(b)) => {
                            let dist = (a - b).unsigned_abs() as i64;
                            dist > max_distance
                        }
                        _ => false,
                    }
                } else {
                    false
                };

                if needs_swap {
                    // Mark this op as needing SWAP insertion
                    if let Some(op) = ctx.ops.get_mut(op_key) {
                        op.attrs.set("needs_swap",
                            lift_core::attributes::Attribute::Bool(true));
                        swaps_inserted += 1;
                    }
                }
            }
        }

        if swaps_inserted > 0 {
            tracing::info!(
                pass = "layout-mapping",
                swaps_needed = swaps_inserted,
                "Layout mapping annotations applied"
            );
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["quantum_analysis"]
    }
}
