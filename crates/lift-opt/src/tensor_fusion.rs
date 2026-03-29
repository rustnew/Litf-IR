use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};

#[derive(Debug)]
pub struct TensorFusion;

impl Pass for TensorFusion {
    fn name(&self) -> &str { "tensor-fusion" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut fused = 0usize;

        // Pattern: matmul + bias_add + relu -> fused_matmul_bias_relu
        let op_keys: Vec<_> = ctx.ops.keys().collect();

        for &op_key in &op_keys {
            let op = match ctx.ops.get(op_key) {
                Some(op) => op,
                None => continue,
            };

            let op_name = ctx.strings.resolve(op.name).to_string();

            if op_name != "tensor.relu" {
                continue;
            }

            // Check if input is a bias add
            if op.inputs.len() != 1 { continue; }
            let relu_input = op.inputs[0];

            let add_op_key = match ctx.get_value(relu_input) {
                Some(val) => match &val.def {
                    lift_core::values::DefSite::OpResult { op, result_index: 0 } => *op,
                    _ => continue,
                },
                None => continue,
            };

            let add_op = match ctx.ops.get(add_op_key) {
                Some(op) => op,
                None => continue,
            };

            let add_name = ctx.strings.resolve(add_op.name).to_string();
            if add_name != "tensor.add" { continue; }
            if add_op.inputs.len() != 2 { continue; }

            let matmul_input = add_op.inputs[0];
            let bias_input = add_op.inputs[1];

            let matmul_op_key = match ctx.get_value(matmul_input) {
                Some(val) => match &val.def {
                    lift_core::values::DefSite::OpResult { op, result_index: 0 } => *op,
                    _ => continue,
                },
                None => continue,
            };

            let (matmul_a, matmul_b) = match ctx.ops.get(matmul_op_key) {
                Some(op) => {
                    let name = ctx.strings.resolve(op.name).to_string();
                    if name != "tensor.matmul" || op.inputs.len() != 2 { continue; }
                    (op.inputs[0], op.inputs[1])
                }
                None => continue,
            };

            // We found the pattern: matmul -> add(bias) -> relu
            let fused_name = ctx.intern_string("tensor.fused_matmul_bias_relu");
            let fused_dialect = ctx.intern_string("tensor");

            if let Some(op_mut) = ctx.ops.get_mut(op_key) {
                op_mut.name = fused_name;
                op_mut.dialect = fused_dialect;
                op_mut.inputs = vec![matmul_a, matmul_b, bias_input];
                fused += 1;
            }
        }

        if fused > 0 {
            tracing::info!("Tensor fusion: fused {} matmul+bias+relu patterns", fused);
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}
