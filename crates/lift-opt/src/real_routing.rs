use lift_core::context::Context;
use lift_core::location::Location;
use lift_core::pass::{AnalysisCache, Pass, PassResult};
use lift_quantum::topology::DeviceTopology;
use std::collections::HashMap;

/// Real layout routing pass.
///
/// Unlike the legacy `LayoutMapping` pass (which only annotates gates with
/// `needs_swap = true`), this pass performs actual routing: for every two-qubit
/// gate whose qubits are not adjacent in the target device topology, it inserts
/// physical SWAP gates along a BFS shortest path so that the logical qubits
/// become adjacent before the gate executes.
///
/// A running `logical -> physical` mapping is maintained as SWAPs modify the
/// placement of logical qubits.
#[derive(Debug)]
pub struct RealRouting {
    topology: DeviceTopology,
}

impl RealRouting {
    pub fn new(topology: DeviceTopology) -> Self {
        Self { topology }
    }
}

impl Default for RealRouting {
    fn default() -> Self {
        Self {
            topology: DeviceTopology::linear(8),
        }
    }
}

impl Pass for RealRouting {
    fn name(&self) -> &str {
        "real-routing"
    }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut swaps_inserted = 0usize;

        // logical -> physical placement (initial: identity).
        let mut placement: HashMap<usize, usize> = HashMap::new();
        let mut reverse: HashMap<usize, usize> = HashMap::new(); // physical -> logical

        // The collect is required, not needless: the loop body mutates `ctx`
        // (inserting SWAP ops and values), which would conflict with an
        // active borrow from an uncollected `ctx.blocks.keys()` iterator.
        #[allow(clippy::needless_collect)]
        let block_keys: Vec<_> = ctx.blocks.keys().collect();
        for block_key in block_keys {
            // We rebuild the block op list as we insert SWAPs, so process in
            // chunks: handle one gate at a time and refresh the snapshot.
            // The `loop + index` form is required because the op list grows
            // while we iterate (we insert SWAPs).
            let mut index = 0usize;
            #[allow(clippy::while_let_loop)]
            loop {
                let current_list = match ctx.blocks.get(block_key) {
                    Some(b) => b.ops.clone(),
                    None => break,
                };
                if index >= current_list.len() {
                    break;
                }
                let op_key = current_list[index];

                let (is_2q, q0_logical, q1_logical, has_qubit_attrs) = {
                    let op = match ctx.ops.get(op_key) {
                        Some(op) => op,
                        None => {
                            index += 1;
                            continue;
                        }
                    };
                    let name = ctx.strings.resolve(op.name);
                    if !name.starts_with("quantum.") || op.inputs.len() < 2 {
                        index += 1;
                        continue;
                    }
                    let q0 = op.attrs.get_integer("qubit0");
                    let q1 = op.attrs.get_integer("qubit1");
                    match (q0, q1) {
                        (Some(a), Some(b)) => (true, a as usize, b as usize, true),
                        _ => (true, 0, 0, false),
                    }
                };

                if !is_2q || !has_qubit_attrs {
                    index += 1;
                    continue;
                }

                // Ensure both logical qubits are in the placement. Initial
                // placement is the identity mapping (logical i -> physical i)
                // when the topology allows it.
                if let std::collections::hash_map::Entry::Vacant(e) = placement.entry(q0_logical) {
                    let phys = if q0_logical < self.topology.num_qubits
                        && !reverse.contains_key(&q0_logical)
                    {
                        q0_logical
                    } else {
                        (0..self.topology.num_qubits)
                            .find(|p| !reverse.contains_key(p))
                            .unwrap_or(q0_logical.min(self.topology.num_qubits.saturating_sub(1)))
                    };
                    e.insert(phys);
                    reverse.insert(phys, q0_logical);
                }
                if let std::collections::hash_map::Entry::Vacant(e) = placement.entry(q1_logical) {
                    let phys = if q1_logical < self.topology.num_qubits
                        && !reverse.contains_key(&q1_logical)
                    {
                        q1_logical
                    } else {
                        (0..self.topology.num_qubits)
                            .find(|p| !reverse.contains_key(p))
                            .unwrap_or(q1_logical.min(self.topology.num_qubits.saturating_sub(1)))
                    };
                    e.insert(phys);
                    reverse.insert(phys, q1_logical);
                }

                let phys0 = placement[&q0_logical];
                let phys1 = placement[&q1_logical];

                if self.topology.are_connected(phys0, phys1) {
                    index += 1;
                    continue;
                }

                // Route: BFS path from phys0 to phys1; insert SWAPs along the
                // path so the qubit at phys0 travels towards phys1.
                let Some(path) = self.topology.shortest_path(phys0, phys1) else {
                    index += 1;
                    continue;
                };

                // path = [phys0, a, b, ..., phys1]. We swap phys0 towards the
                // first step, updating placement each time.
                let mut current_phys = phys0;
                for &next in &path[1..] {
                    // The inputs to the original gate (snapshot so we can mutate
                    // ctx below).
                    let op_inputs = match ctx.ops.get(op_key) {
                        Some(o) => o.inputs.clone(),
                        None => break,
                    };
                    if op_inputs.len() < 2 {
                        break;
                    }

                    // The other logical qubit currently living at physical `next`.
                    let other_logical = reverse.get(&next).copied();

                    let qubit_ty = ctx.make_qubit_type();

                    let swap_inputs = if current_phys == phys0 {
                        // We are moving q0_logical: swap its value with whatever
                        // is at `next` (or a fresh ancilla).
                        let routed_val = op_inputs[0];
                        let target_val = match other_logical {
                            Some(l) if l == q1_logical => op_inputs[1],
                            _ => {
                                // An empty physical qubit: introduce an ancilla
                                // qubit value (block arg would be cleaner, but a
                                // fresh value keeps the gate well-formed).
                                ctx.create_value(
                                    qubit_ty,
                                    None,
                                    lift_core::values::DefSite::BlockArg {
                                        block: block_key,
                                        arg_index: 0,
                                    },
                                )
                            }
                        };
                        vec![routed_val, target_val]
                    } else {
                        vec![op_inputs[0], op_inputs[1]]
                    };

                    let (swap_op, swap_results) = ctx.create_op(
                        "quantum.swap",
                        "quantum",
                        swap_inputs,
                        vec![qubit_ty, qubit_ty],
                        lift_core::attributes::Attributes::new(),
                        Location::unknown(),
                    );
                    ctx.insert_op_before(op_key, swap_op);

                    // Update the gate's inputs: for the moved qubit, use the
                    // first SWAP result.
                    if let Some(op) = ctx.ops.get_mut(op_key) {
                        if current_phys == phys0 {
                            op.inputs[0] = swap_results[0];
                        } else {
                            op.inputs[1] = swap_results[1];
                        }
                    }

                    // Update placement: the logical qubit at `next` (if any)
                    // moves to current_phys, and q0_logical moves to next.
                    let swapped_logical = reverse.get(&next).copied();
                    if let Some(l) = swapped_logical {
                        placement.insert(l, current_phys);
                        reverse.insert(current_phys, l);
                    }
                    placement.insert(q0_logical, next);
                    reverse.insert(next, q0_logical);
                    reverse.remove(&current_phys);

                    current_phys = next;
                    swaps_inserted += 1;
                }

                // After routing, the gate is well-placed; continue after it.
                index += 1;
            }
        }

        if swaps_inserted > 0 {
            tracing::info!(
                pass = "real-routing",
                swaps_inserted = swaps_inserted,
                "Physical SWAPs inserted for layout routing"
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
    use lift_core::values::ValueKey;

    fn build_ctx() -> (Context, lift_core::blocks::BlockKey, Vec<ValueKey>) {
        let mut ctx = Context::new();
        let qubit = ctx.make_qubit_type();
        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit);
        let q1 = ctx.create_block_arg(block, qubit);
        (ctx, block, vec![q0, q1])
    }

    /// CX between logical 0 and 3 on a 4-qubit linear topology (0-1-2-3).
    /// Distance 3 -> needs 2 SWAPs to bring 0 next to 3.
    #[test]
    fn test_inserts_swaps_for_distant_gate() {
        let (mut ctx, block, qubits) = build_ctx();
        let qty = ctx.make_qubit_type();
        let mut attrs = Attributes::new();
        attrs.set("qubit0", Attribute::Integer(0));
        attrs.set("qubit1", Attribute::Integer(3));
        let (cx, _) = ctx.create_op(
            "quantum.cx",
            "quantum",
            qubits,
            vec![qty, qty],
            attrs,
            Location::unknown(),
        );
        ctx.add_op_to_block(block, cx);

        let topology = DeviceTopology::linear(4);
        let pass = RealRouting::new(topology);
        let result = pass.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        let swap_count = ctx
            .ops
            .values()
            .filter(|op| ctx.strings.resolve(op.name) == "quantum.swap")
            .count();
        assert!(swap_count >= 2, "expected >=2 SWAPs, got {}", swap_count);
    }

    /// CX between adjacent qubits 0-1 -> no SWAPs.
    #[test]
    fn test_no_swaps_for_adjacent_gate() {
        let (mut ctx, block, qubits) = build_ctx();
        let qty = ctx.make_qubit_type();
        let mut attrs = Attributes::new();
        attrs.set("qubit0", Attribute::Integer(0));
        attrs.set("qubit1", Attribute::Integer(1));
        let (cx, _) = ctx.create_op(
            "quantum.cx",
            "quantum",
            qubits,
            vec![qty, qty],
            attrs,
            Location::unknown(),
        );
        ctx.add_op_to_block(block, cx);

        let topology = DeviceTopology::linear(4);
        let pass = RealRouting::new(topology);
        let result = pass.run(&mut ctx, &mut AnalysisCache::new());
        assert_eq!(result, PassResult::Unchanged);
    }
}
