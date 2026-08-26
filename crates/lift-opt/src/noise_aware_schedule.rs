use lift_core::context::Context;
use lift_core::pass::{AnalysisCache, Pass, PassResult};

/// Noise-aware scheduling pass: reorders quantum gates to minimise
/// decoherence by scheduling operations on qubits with longer T1/T2
/// times first, and minimising idle time on noisy qubits.
#[derive(Debug)]
pub struct NoiseAwareSchedule;

impl Pass for NoiseAwareSchedule {
    fn name(&self) -> &str {
        "noise-aware-schedule"
    }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut reordered = 0usize;

        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let op_list = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            if op_list.len() < 2 {
                continue;
            }

            let quantum_op_count = op_list
                .iter()
                .filter(|&&op_key| {
                    ctx.ops
                        .get(op_key)
                        .is_some_and(|op| ctx.strings.resolve(op.name).starts_with("quantum."))
                })
                .count();

            if quantum_op_count < 2 {
                continue;
            }

            // Sort quantum ops: shorter gate times first to reduce idle time,
            // grouped into runs that are safe to reorder among themselves.
            // A non-quantum op is a hard group boundary — quantum ops on
            // either side of it must not be reordered across it, since it may
            // consume a preceding quantum op's result (e.g. `core.return`) or
            // otherwise depend on program order. Data dependencies between
            // quantum ops (via SSA inputs) are the other boundary.
            let mut independent_groups: Vec<Vec<(lift_core::operations::OpKey, f64)>> = Vec::new();
            let mut current_group: Vec<(lift_core::operations::OpKey, f64)> = Vec::new();

            for &op_key in &op_list {
                let is_quantum = ctx
                    .ops
                    .get(op_key)
                    .is_some_and(|op| ctx.strings.resolve(op.name).starts_with("quantum."));

                if !is_quantum {
                    if !current_group.is_empty() {
                        independent_groups.push(current_group.clone());
                        current_group.clear();
                    }
                    continue;
                }

                let gate_time = ctx
                    .ops
                    .get(op_key)
                    .and_then(|op| op.attrs.get_float("gate_time_us"))
                    .unwrap_or(0.1);

                let depends_on_prev = if let Some(op) = ctx.ops.get(op_key) {
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
                current_group.push((op_key, gate_time));
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
                if orig != sorted {
                    reordered += 1;
                }
                new_quantum_order.extend(group.iter().map(|(k, _)| *k));
            }

            // Rebuild block ops in original program order, substituting the
            // reordered quantum ops into the exact slots quantum ops
            // occupied. Non-quantum ops (e.g. `core.return`) keep their
            // original position — they used to be unconditionally hoisted
            // after every quantum op, which reordered a return past the
            // gates producing the values it returns.
            if reordered > 0 {
                if let Some(block) = ctx.blocks.get_mut(block_key) {
                    let mut new_quantum_iter = new_quantum_order.into_iter();
                    let new_ops: Vec<_> = op_list
                        .iter()
                        .map(|&op_key| {
                            let is_quantum = ctx.ops.get(op_key).is_some_and(|op| {
                                ctx.strings.resolve(op.name).starts_with("quantum.")
                            });
                            if is_quantum {
                                new_quantum_iter.next().expect("one slot per quantum op")
                            } else {
                                op_key
                            }
                        })
                        .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::attributes::{Attribute, Attributes};
    use lift_core::location::Location;

    /// Regression test: H(t=1.0) and Z(t=0.5) are independent (different
    /// qubits, no data dependency) so the pass reorders them by gate time.
    /// `core.return` consumes the CX that consumes both gates' outputs, and
    /// must stay last — it used to be unconditionally hoisted before every
    /// quantum op, which would move it ahead of the gates producing its
    /// operands.
    #[test]
    fn test_return_stays_after_the_gates_it_consumes() {
        let mut ctx = Context::new();
        let qubit = ctx.make_qubit_type();
        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit);
        let q1 = ctx.create_block_arg(block, qubit);

        let mut h_attrs = Attributes::new();
        h_attrs.set("gate_time_us", Attribute::Float(1.0));
        let (h, h_res) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![q0],
            vec![qubit],
            h_attrs,
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h);

        let mut z_attrs = Attributes::new();
        z_attrs.set("gate_time_us", Attribute::Float(0.5));
        let (z, z_res) = ctx.create_op(
            "quantum.z",
            "quantum",
            vec![q1],
            vec![qubit],
            z_attrs,
            Location::unknown(),
        );
        ctx.add_op_to_block(block, z);

        let (cx, cx_res) = ctx.create_op(
            "quantum.cx",
            "quantum",
            vec![h_res[0], z_res[0]],
            vec![qubit, qubit],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, cx);

        let (ret, _) = ctx.create_op(
            "core.return",
            "core",
            cx_res.clone(),
            vec![],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, ret);

        let result = NoiseAwareSchedule.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed(), "H and Z should be reordered by gate time");

        let final_block = ctx.blocks.get(block).unwrap();
        let names: Vec<String> = final_block
            .ops
            .iter()
            .map(|&k| {
                ctx.strings
                    .resolve(ctx.ops.get(k).unwrap().name)
                    .to_string()
            })
            .collect();

        assert_eq!(
            names.last().map(String::as_str),
            Some("core.return"),
            "core.return must stay last, not be hoisted before the gates it consumes: {:?}",
            names
        );
        // Z (shorter gate time) should now come before H.
        assert_eq!(
            names[0], "quantum.z",
            "shorter gate time should be scheduled first: {:?}",
            names
        );
    }
}
