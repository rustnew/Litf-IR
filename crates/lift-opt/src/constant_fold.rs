use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use lift_core::attributes::Attribute;

#[derive(Debug)]
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn name(&self) -> &str { "constant-folding" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut folded = 0usize;

        // Collect ops that can be folded
        let op_keys: Vec<_> = ctx.ops.keys().collect();

        for op_key in op_keys {
            let op = match ctx.ops.get(op_key) {
                Some(op) => op,
                None => continue,
            };

            let op_name = ctx.strings.resolve(op.name).to_string();

            // Check if all inputs are constants
            let all_constant = op.inputs.iter().all(|&input| {
                if let Some(val) = ctx.get_value(input) {
                    if let Some(def_op) = match &val.def {
                        lift_core::values::DefSite::OpResult { op, .. } => ctx.get_op(*op),
                        _ => None,
                    } {
                        let def_name = ctx.strings.resolve(def_op.name);
                        return def_name == "core.constant";
                    }
                }
                false
            });

            if !all_constant || op.inputs.is_empty() {
                continue;
            }

            // Get constant values
            let const_values: Vec<Option<&Attribute>> = op.inputs.iter().map(|&input| {
                let val = ctx.get_value(input)?;
                let def_op = match &val.def {
                    lift_core::values::DefSite::OpResult { op, .. } => ctx.get_op(*op),
                    _ => None,
                }?;
                def_op.attrs.get("value")
            }).collect();

            // Try to fold based on operation
            let folded_value = match op_name.as_str() {
                "tensor.add" if const_values.len() == 2 => {
                    fold_binary_int(&const_values, |a, b| a + b)
                        .or_else(|| fold_binary_float(&const_values, |a, b| a + b))
                }
                "tensor.sub" if const_values.len() == 2 => {
                    fold_binary_int(&const_values, |a, b| a - b)
                        .or_else(|| fold_binary_float(&const_values, |a, b| a - b))
                }
                "tensor.mul" if const_values.len() == 2 => {
                    fold_binary_int(&const_values, |a, b| a * b)
                        .or_else(|| fold_binary_float(&const_values, |a, b| a * b))
                }
                "tensor.neg" if const_values.len() == 1 => {
                    match const_values[0] {
                        Some(Attribute::Integer(v)) => Some(Attribute::Integer(-v)),
                        Some(Attribute::Float(v)) => Some(Attribute::Float(-v)),
                        _ => None,
                    }
                }
                _ => None,
            };

            if let Some(new_val) = folded_value {
                // Intern strings before borrowing ops mutably
                let const_name = ctx.intern_string("core.constant");
                let const_dialect = ctx.intern_string("core");
                if let Some(op_mut) = ctx.ops.get_mut(op_key) {
                    op_mut.name = const_name;
                    op_mut.dialect = const_dialect;
                    op_mut.inputs.clear();
                    op_mut.attrs.set("value", new_val);
                    folded += 1;
                }
            }
        }

        if folded > 0 {
            tracing::info!("Constant folding: folded {} operations", folded);
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}

fn fold_binary_int(values: &[Option<&Attribute>], f: impl Fn(i64, i64) -> i64) -> Option<Attribute> {
    match (values.first()?, values.get(1)?) {
        (Some(Attribute::Integer(a)), Some(Attribute::Integer(b))) => {
            Some(Attribute::Integer(f(*a, *b)))
        }
        _ => None,
    }
}

fn fold_binary_float(values: &[Option<&Attribute>], f: impl Fn(f64, f64) -> f64) -> Option<Attribute> {
    match (values.first()?, values.get(1)?) {
        (Some(Attribute::Float(a)), Some(Attribute::Float(b))) => {
            Some(Attribute::Float(f(*a, *b)))
        }
        _ => None,
    }
}
