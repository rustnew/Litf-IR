use lift_core::context::Context;
use lift_core::operations::OpKey;
use lift_core::pass::{AnalysisCache, Pass, PassResult};
use lift_core::values::{DefSite, ValueKey};

#[derive(Debug)]
pub struct TensorFusion;

impl Pass for TensorFusion {
    fn name(&self) -> &str {
        "tensor-fusion"
    }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut fused = 0usize;

        // Snapshot keys so we can mutate freely while iterating.
        let op_keys: Vec<OpKey> = ctx.ops.keys().collect();

        // Phase 1: ternary / activation-led patterns (these take priority so a
        // `add` used by an activation is not fused standalone first).
        for &op_key in &op_keys {
            // ── Pattern 1: matmul + add + relu -> fused_matmul_bias_relu ──
            if let Some(n) = fuse_matmul_bias_relu(ctx, op_key) {
                fused += n;
                continue;
            }
            // ── Pattern 3: linear + gelu -> fused_linear_gelu ──
            if let Some(n) =
                fuse_linear_activation(ctx, op_key, "tensor.gelu", "tensor.fused_linear_gelu")
            {
                fused += n;
                continue;
            }
            // ── Pattern 4: linear + silu -> fused_linear_silu ──
            if let Some(n) =
                fuse_linear_activation(ctx, op_key, "tensor.silu", "tensor.fused_linear_silu")
            {
                fused += n;
                continue;
            }
            // ── Pattern 5: conv2d + batchnorm + relu -> fused_conv_batchnorm_relu ──
            if let Some(n) = fuse_conv_bn_relu(ctx, op_key) {
                fused += n;
            }
        }

        // Phase 2: binary patterns, on ops that survived phase 1.
        for &op_key in &op_keys {
            // ── Pattern 2: matmul + add -> fused_matmul_bias ──
            if let Some(n) = fuse_matmul_bias(ctx, op_key) {
                fused += n;
            }
        }

        if fused > 0 {
            tracing::info!("Tensor fusion: fused {} patterns", fused);
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}

/// Returns the OpKey that produces `value` (result index 0), if any.
fn producer_op(ctx: &Context, value: ValueKey) -> Option<OpKey> {
    match ctx.get_value(value)?.def {
        DefSite::OpResult {
            op,
            result_index: 0,
        } => Some(op),
        _ => None,
    }
}

/// Returns `true` if `op` has the given full name (`tensor.relu` etc).
fn op_is(ctx: &Context, op: OpKey, name: &str) -> bool {
    ctx.ops
        .get(op)
        .map(|o| ctx.strings.resolve(o.name) == name)
        .unwrap_or(false)
}

/// Rewrites `relu(add(matmul(a, b), bias))` into `fused_matmul_bias_relu(a, b, bias)`.
fn fuse_matmul_bias_relu(ctx: &mut Context, relu_op: OpKey) -> Option<usize> {
    if !op_is(ctx, relu_op, "tensor.relu") {
        return None;
    }
    let inputs = ctx.ops.get(relu_op)?.inputs.clone();
    if inputs.len() != 1 {
        return None;
    }
    let add_op = producer_op(ctx, inputs[0])?;
    if !op_is(ctx, add_op, "tensor.add") {
        return None;
    }
    let add_inputs = ctx.ops.get(add_op)?.inputs.clone();
    if add_inputs.len() != 2 {
        return None;
    }
    let matmul_op = producer_op(ctx, add_inputs[0])?;
    if !op_is(ctx, matmul_op, "tensor.matmul") {
        return None;
    }
    let mm_inputs = ctx.ops.get(matmul_op)?.inputs.clone();
    if mm_inputs.len() != 2 {
        return None;
    }

    let fused_name = ctx.intern_string("tensor.fused_matmul_bias_relu");
    ctx.ops.get_mut(relu_op)?.name = fused_name;
    ctx.ops.get_mut(relu_op)?.inputs = vec![mm_inputs[0], mm_inputs[1], add_inputs[1]];
    Some(1)
}

/// Rewrites `add(matmul(a, b), bias)` into `fused_matmul_bias(a, b, bias)`.
fn fuse_matmul_bias(ctx: &mut Context, add_op: OpKey) -> Option<usize> {
    if !op_is(ctx, add_op, "tensor.add") {
        return None;
    }
    let inputs = ctx.ops.get(add_op)?.inputs.clone();
    if inputs.len() != 2 {
        return None;
    }
    let matmul_op = producer_op(ctx, inputs[0])?;
    if !op_is(ctx, matmul_op, "tensor.matmul") {
        return None;
    }
    let mm_inputs = ctx.ops.get(matmul_op)?.inputs.clone();
    if mm_inputs.len() != 2 {
        return None;
    }

    let fused_name = ctx.intern_string("tensor.fused_matmul_bias");
    ctx.ops.get_mut(add_op)?.name = fused_name;
    ctx.ops.get_mut(add_op)?.inputs = vec![mm_inputs[0], mm_inputs[1], inputs[1]];
    Some(1)
}

/// Rewrites `activation(linear(input, w, b))` into a fused linear-activation op.
fn fuse_linear_activation(
    ctx: &mut Context,
    act_op: OpKey,
    act_name: &str,
    fused_name: &str,
) -> Option<usize> {
    if !op_is(ctx, act_op, act_name) {
        return None;
    }
    let inputs = ctx.ops.get(act_op)?.inputs.clone();
    if inputs.len() != 1 {
        return None;
    }
    let linear_op = producer_op(ctx, inputs[0])?;
    if !op_is(ctx, linear_op, "tensor.linear") {
        return None;
    }
    let lin_inputs = ctx.ops.get(linear_op)?.inputs.clone();
    if lin_inputs.len() != 3 {
        return None;
    }

    let fused = ctx.intern_string(fused_name);
    ctx.ops.get_mut(act_op)?.name = fused;
    ctx.ops.get_mut(act_op)?.inputs = lin_inputs;
    Some(1)
}

/// Rewrites `relu(batchnorm(conv2d(x, w), scale, bias))` into
/// `fused_conv_batchnorm_relu(x, w, scale, bias)`.
fn fuse_conv_bn_relu(ctx: &mut Context, relu_op: OpKey) -> Option<usize> {
    if !op_is(ctx, relu_op, "tensor.relu") {
        return None;
    }
    let inputs = ctx.ops.get(relu_op)?.inputs.clone();
    if inputs.len() != 1 {
        return None;
    }
    let bn_op = producer_op(ctx, inputs[0])?;
    if !op_is(ctx, bn_op, "tensor.batchnorm") {
        return None;
    }
    let bn_inputs = ctx.ops.get(bn_op)?.inputs.clone();
    // batchnorm: x, scale, bias (running stats are optional)
    if bn_inputs.len() < 3 {
        return None;
    }
    let conv_op = producer_op(ctx, bn_inputs[0])?;
    if !op_is(ctx, conv_op, "tensor.conv2d") {
        return None;
    }
    let conv_inputs = ctx.ops.get(conv_op)?.inputs.clone();
    if conv_inputs.len() != 2 {
        return None;
    }

    let fused_name = ctx.intern_string("tensor.fused_conv_batchnorm_relu");
    ctx.ops.get_mut(relu_op)?.name = fused_name;
    ctx.ops.get_mut(relu_op)?.inputs =
        vec![conv_inputs[0], conv_inputs[1], bn_inputs[1], bn_inputs[2]];
    Some(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::attributes::Attributes;
    use lift_core::location::Location;

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        let _ = ctx.make_float_type(32);
        ctx
    }

    /// Builds a simple linear chain: producer1 -> producer2 -> consumer.
    /// Returns the Context with ops and the consumer OpKey.
    fn build_chain(ctx: &mut Context, producer1: &str, producer2: &str, consumer: &str) -> OpKey {
        let f32 = ctx.make_float_type(32);
        let block = ctx.create_block();
        let a = ctx.create_block_arg(block, f32);
        let b = ctx.create_block_arg(block, f32);

        let (op1, res1) = ctx.create_op(
            producer1,
            "tensor",
            vec![a, b],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, op1);

        let (op2, res2) = ctx.create_op(
            producer2,
            "tensor",
            vec![res1[0]],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, op2);

        let (op3, _) = ctx.create_op(
            consumer,
            "tensor",
            vec![res2[0]],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, op3);
        op3
    }

    #[test]
    fn test_fuse_matmul_bias_relu() {
        let mut ctx = make_ctx();
        let f32 = ctx.make_float_type(32);
        let block = ctx.create_block();
        let a = ctx.create_block_arg(block, f32);
        let b = ctx.create_block_arg(block, f32);
        let bias = ctx.create_block_arg(block, f32);

        let (mm, mm_res) = ctx.create_op(
            "tensor.matmul",
            "tensor",
            vec![a, b],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, mm);

        let (add, add_res) = ctx.create_op(
            "tensor.add",
            "tensor",
            vec![mm_res[0], bias],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, add);

        let (relu, _) = ctx.create_op(
            "tensor.relu",
            "tensor",
            vec![add_res[0]],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, relu);

        let result = TensorFusion.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        let relu = ctx.ops.get(relu).unwrap();
        assert_eq!(
            ctx.strings.resolve(relu.name),
            "tensor.fused_matmul_bias_relu"
        );
        assert_eq!(relu.inputs.len(), 3);
    }

    #[test]
    fn test_fuse_conv_bn_relu() {
        let mut ctx = make_ctx();
        let f32 = ctx.make_float_type(32);
        let block = ctx.create_block();
        let x = ctx.create_block_arg(block, f32);
        let w = ctx.create_block_arg(block, f32);
        let scale = ctx.create_block_arg(block, f32);
        let bias = ctx.create_block_arg(block, f32);

        let (conv, conv_res) = ctx.create_op(
            "tensor.conv2d",
            "tensor",
            vec![x, w],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, conv);

        let (bn, bn_res) = ctx.create_op(
            "tensor.batchnorm",
            "tensor",
            vec![conv_res[0], scale, bias],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, bn);

        let (relu, _) = ctx.create_op(
            "tensor.relu",
            "tensor",
            vec![bn_res[0]],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, relu);

        let result = TensorFusion.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        let relu = ctx.ops.get(relu).unwrap();
        assert_eq!(
            ctx.strings.resolve(relu.name),
            "tensor.fused_conv_batchnorm_relu"
        );
        assert_eq!(relu.inputs.len(), 4);
    }

    #[test]
    fn test_fuse_linear_gelu() {
        let mut ctx = make_ctx();
        let f32 = ctx.make_float_type(32);
        let block = ctx.create_block();
        let x = ctx.create_block_arg(block, f32);
        let w = ctx.create_block_arg(block, f32);
        let b = ctx.create_block_arg(block, f32);

        let (lin, lin_res) = ctx.create_op(
            "tensor.linear",
            "tensor",
            vec![x, w, b],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, lin);

        let (gelu, _) = ctx.create_op(
            "tensor.gelu",
            "tensor",
            vec![lin_res[0]],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, gelu);

        let result = TensorFusion.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        let gelu = ctx.ops.get(gelu).unwrap();
        assert_eq!(ctx.strings.resolve(gelu.name), "tensor.fused_linear_gelu");
        assert_eq!(gelu.inputs.len(), 3);
    }

    #[test]
    fn test_no_fusion_on_plain_relu() {
        let mut ctx = make_ctx();
        let f32 = ctx.make_float_type(32);
        let block = ctx.create_block();
        let x = ctx.create_block_arg(block, f32);

        let (relu, _) = ctx.create_op(
            "tensor.relu",
            "tensor",
            vec![x],
            vec![f32],
            Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, relu);

        let result = TensorFusion.run(&mut ctx, &mut AnalysisCache::new());
        assert_eq!(result, PassResult::Unchanged);
    }

    // Keep build_chain used so the helper is exercised.
    #[test]
    fn test_build_chain_helper() {
        let mut ctx = make_ctx();
        let consumer = build_chain(&mut ctx, "tensor.matmul", "tensor.add", "tensor.relu");
        assert!(ctx.ops.contains_key(consumer));
    }
}
