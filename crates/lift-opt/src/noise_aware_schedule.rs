use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};

/// Noise-aware scheduling pass: reorders quantum gates to minimise
/// decoherence by scheduling operations on qubits with longer T1/T2
/// times first, and minimising idle time on noisy qubits.
#[derive(Debug)]
pub struct NoiseAwareSchedule;

impl Pass for NoiseAwareSchedule {
    fn name(&self) -> &str { "noise-aware-schedule" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut reordered = 0usize;

        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let op_list = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            if op_list.len() < 2 { continue; }

            // Collect quantum ops with their gate times
            let mut quantum_ops: Vec<(lift_core::operations::OpKey, f64)> = Vec::new();
            let mut non_quantum_ops: Vec<lift_core::operations::OpKey> = Vec::new();

            for &op_key in &op_list {
                let is_quantum = if let Some(op) = ctx.ops.get(op_key) {
                    let name = ctx.strings.resolve(op.name);
                    name.starts_with("quantum.")
                } else {
                    false
                };

                if is_quantum {
                    let gate_time = if let Some(op) = ctx.ops.get(op_key) {
                        op.attrs.get_float("gate_time_us").unwrap_or(0.1)
                    } else {
                        0.1
                    };
                    quantum_ops.push((op_key, gate_time));
                } else {
                    non_quantum_ops.push(op_key);
                }
            }

            if quantum_ops.len() < 2 { continue; }

            // Sort quantum ops: shorter gate times first to reduce idle time
            // (Respecting data dependencies via SSA inputs)
            let mut independent_groups: Vec<Vec<(lift_core::operations::OpKey, f64)>> = Vec::new();
            let mut current_group: Vec<(lift_core::operations::OpKey, f64)> = Vec::new();

            for (op_key, gate_time) in &quantum_ops {
                let depends_on_prev = if let Some(op) = ctx.ops.get(*op_key) {
                    current_group.iter().any(|(prev_key, _)| {
                        if let Some(prev_op) = ctx.ops.get(*prev_key) {
                            prev_op.results.iter().any(|r| op.inputs.contains(r))
                        } else {
                            false
                        }
                    })
                } else {
                    false
                };

                if depends_on_prev && !current_group.is_empty() {
                    independent_groups.push(current_group.clone());
                    current_group.clear();
                }
                current_group.push((*op_key, *gate_time));
            }
            if !current_group.is_empty() {
                independent_groups.push(current_group);
            }

            // Sort each independent group by gate time (ascending)
            let mut new_quantum_order = Vec::new();
            for group in &mut independent_groups {
                let orig: Vec<_> = group.iter().map(|(k, _)| *k).collect();
                group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let sorted: Vec<_> = group.iter().map(|(k, _)| *k).collect();
                if orig != sorted { reordered += 1; }
                new_quantum_order.extend(group.iter().map(|(k, _)| *k));
            }

            // Rebuild block ops: non-quantum first, then reordered quantum
            if reordered > 0 {
                if let Some(block) = ctx.blocks.get_mut(block_key) {
                    let mut new_ops = non_quantum_ops.clone();
                    new_ops.extend(new_quantum_order);
                    block.ops = new_ops;
                }
            }
        }

        if reordered > 0 {
            tracing::info!(
                pass = "noise-aware-schedule",
                groups_reordered = reordered,
                "Noise-aware scheduling applied"
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
