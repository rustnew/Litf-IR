use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use lift_core::attributes::Attribute;

#[derive(Debug)]
pub struct Canonicalize;

impl Pass for Canonicalize {
    fn name(&self) -> &str { "canonicalize" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut changed = false;

        let op_keys: Vec<_> = ctx.ops.keys().collect();

        for op_key in op_keys {
            let op = match ctx.ops.get(op_key) {
                Some(op) => op,
                None => continue,
            };

            let op_name = ctx.strings.resolve(op.name).to_string();

            match op_name.as_str() {
                // x + 0 -> x
                "tensor.add" => {
                    if op.inputs.len() == 2 {
                        if is_zero_constant(ctx, op.inputs[1]) {
                            let input = op.inputs[0];
                            let result = op.results[0];
                            rewire_value(ctx, result, input, op_key);
                            changed = true;
                        } else if is_zero_constant(ctx, op.inputs[0]) {
                            let input = op.inputs[1];
                            let result = op.results[0];
                            rewire_value(ctx, result, input, op_key);
                            changed = true;
                        }
                    }
                }
                // x * 1 -> x
                "tensor.mul" => {
                    if op.inputs.len() == 2 {
                        if is_one_constant(ctx, op.inputs[1]) {
                            let input = op.inputs[0];
                            let result = op.results[0];
                            rewire_value(ctx, result, input, op_key);
                            changed = true;
                        } else if is_one_constant(ctx, op.inputs[0]) {
                            let input = op.inputs[1];
                            let result = op.results[0];
                            rewire_value(ctx, result, input, op_key);
                            changed = true;
                        }
                    }
                }
                // reshape(reshape(x)) -> reshape(x)
                "tensor.reshape" => {
                    if op.inputs.len() == 1 {
                        if let Some(val) = ctx.get_value(op.inputs[0]) {
                            if let lift_core::values::DefSite::OpResult { op: prev_op, .. } = &val.def {
                                if let Some(prev) = ctx.get_op(*prev_op) {
                                    let prev_name = ctx.strings.resolve(prev.name);
                                    if prev_name == "tensor.reshape" && !prev.inputs.is_empty() {
                                        let original_input = prev.inputs[0];
                                        if let Some(op_mut) = ctx.ops.get_mut(op_key) {
                                            op_mut.inputs = vec![original_input];
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if changed {
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }
}

fn is_zero_constant(ctx: &Context, val_key: lift_core::values::ValueKey) -> bool {
    if let Some(val) = ctx.get_value(val_key) {
        if let lift_core::values::DefSite::OpResult { op, .. } = &val.def {
            if let Some(op_data) = ctx.get_op(*op) {
                let name = ctx.strings.resolve(op_data.name);
                if name == "core.constant" {
                    return match op_data.attrs.get("value") {
                        Some(Attribute::Integer(0)) => true,
                        Some(Attribute::Float(v)) => *v == 0.0,
                        _ => false,
                    };
                }
            }
        }
    }
    false
}

fn is_one_constant(ctx: &Context, val_key: lift_core::values::ValueKey) -> bool {
    if let Some(val) = ctx.get_value(val_key) {
        if let lift_core::values::DefSite::OpResult { op, .. } = &val.def {
            if let Some(op_data) = ctx.get_op(*op) {
                let name = ctx.strings.resolve(op_data.name);
                if name == "core.constant" {
                    return match op_data.attrs.get("value") {
                        Some(Attribute::Integer(1)) => true,
                        Some(Attribute::Float(v)) => *v == 1.0,
                        _ => false,
                    };
                }
            }
        }
    }
    false
}

fn rewire_value(
    ctx: &mut Context,
    old_val: lift_core::values::ValueKey,
    new_val: lift_core::values::ValueKey,
    _skip_op: lift_core::operations::OpKey,
) {
    let op_keys: Vec<_> = ctx.ops.keys().collect();
    for ok in op_keys {
        if let Some(op) = ctx.ops.get_mut(ok) {
            for input in &mut op.inputs {
                if *input == old_val {
                    *input = new_val;
                }
            }
        }
    }
}
