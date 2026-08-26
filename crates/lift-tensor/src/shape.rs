use crate::ops::TensorOp;
use lift_core::attributes::Attributes;
use lift_core::types::{Dimension, TensorTypeInfo};

#[derive(Debug)]
pub struct ShapeInference;

/// Reads a spatial conv/pool parameter (`stride`, `padding`, `dilation`) from
/// `attrs`, falling back to `default` when absent. A single integer applies
/// to every spatial axis (symmetric kernels only — matching every example
/// and test in this codebase, none of which use per-axis values).
fn spatial_param(attrs: Option<&Attributes>, key: &str, default: i64) -> i64 {
    attrs.and_then(|a| a.get_integer(key)).unwrap_or(default)
}

/// Standard convolution/pooling output-length formula:
/// `floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`.
fn conv_output_dim(input: i64, kernel: i64, stride: i64, padding: i64, dilation: i64) -> i64 {
    let numerator = input + 2 * padding - dilation * (kernel - 1) - 1;
    (numerator.max(0) / stride.max(1)) + 1
}

impl ShapeInference {
    pub fn infer_output_shape(
        op: &TensorOp,
        inputs: &[&TensorTypeInfo],
        attrs: Option<&Attributes>,
    ) -> Result<Vec<TensorTypeInfo>, String> {
        match op {
            // ── Binary element-wise (broadcast) ──
            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => {
                if inputs.len() != 2 {
                    return Err(format!("{} requires 2 inputs", op.name()));
                }
                let result = broadcast_shapes(&inputs[0].shape, &inputs[1].shape)?;
                Ok(vec![TensorTypeInfo {
                    shape: result,
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Unary shape-preserving ──
            TensorOp::Neg
            | TensorOp::ReLU
            | TensorOp::GeLU
            | TensorOp::SiLU
            | TensorOp::Sigmoid
            | TensorOp::Tanh
            | TensorOp::LeakyReLU
            | TensorOp::ELU
            | TensorOp::Mish
            | TensorOp::HardSwish
            | TensorOp::HardSigmoid
            | TensorOp::Softmax
            | TensorOp::Cumsum
            | TensorOp::Quantize
            | TensorOp::Dequantize
            | TensorOp::QuantizeInt4
            | TensorOp::DequantizeInt4
            | TensorOp::QuantizeFp8
            | TensorOp::DequantizeFp8
            | TensorOp::Checkpoint
            | TensorOp::Offload
            | TensorOp::GradReLU
            | TensorOp::GradGeLU
            | TensorOp::GradSoftmax => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── Normalisation (shape-preserving) ──
            TensorOp::LayerNorm
            | TensorOp::RMSNorm
            | TensorOp::BatchNorm
            | TensorOp::GroupNorm
            | TensorOp::InstanceNorm
            | TensorOp::GradLayerNorm => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── MatMul ──
            TensorOp::MatMul | TensorOp::SparseMatMul => {
                if inputs.len() != 2 {
                    return Err("matmul requires 2 inputs".into());
                }
                let a = &inputs[0].shape;
                let b = &inputs[1].shape;
                if a.len() < 2 || b.len() < 2 {
                    return Err("matmul inputs must be at least 2D".into());
                }
                let m = a[a.len() - 2].clone();
                let n = b[b.len() - 1].clone();

                let k_a = &a[a.len() - 1];
                let k_b = &b[b.len() - 2];
                if let (Some(ka), Some(kb)) = (k_a.static_value(), k_b.static_value()) {
                    if ka != kb {
                        return Err(format!("matmul inner dimension mismatch: {} vs {}", ka, kb));
                    }
                }

                let mut result_shape = Vec::new();
                let batch_a = &a[..a.len() - 2];
                let batch_b = &b[..b.len() - 2];
                let batch = broadcast_shapes(batch_a, batch_b)?;
                result_shape.extend(batch);
                result_shape.push(m);
                result_shape.push(n);

                Ok(vec![TensorTypeInfo {
                    shape: result_shape,
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Linear ──
            TensorOp::Linear => {
                if inputs.len() < 2 {
                    return Err("linear requires at least 2 inputs (x, W)".into());
                }
                let x = &inputs[0].shape;
                let w = &inputs[1].shape;
                if x.is_empty() || w.len() != 2 {
                    return Err("linear: x must be at least 1D, W must be 2D".into());
                }
                let mut result_shape = x[..x.len() - 1].to_vec();
                result_shape.push(w[1].clone());

                Ok(vec![TensorTypeInfo {
                    shape: result_shape,
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Conv2D ──
            TensorOp::Conv2D | TensorOp::DepthwiseConv2D | TensorOp::DilatedConv2D => {
                if inputs.len() < 2 {
                    return Err("conv2d requires at least 2 inputs (input, kernel)".into());
                }
                let input = &inputs[0].shape;
                let kernel = &inputs[1].shape;
                if input.len() != 4 || kernel.len() != 4 {
                    return Err("conv2d: input and kernel must be 4D (NCHW)".into());
                }

                // `quantum.dilated_conv2d`'s dilation and any non-default
                // stride/padding come from attrs — without them (the common
                // case in tests), this reduces to the stride=1/padding=0/
                // dilation=1 formula `in - kernel + 1`.
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(
                    attrs,
                    "dilation",
                    if matches!(op, TensorOp::DilatedConv2D) {
                        2
                    } else {
                        1
                    },
                );

                let n = input[0].clone();
                let cout = kernel[0].clone();
                let h_out = match (&input[2], &kernel[2]) {
                    (Dimension::Constant(ih), Dimension::Constant(kh)) => Dimension::Constant(
                        conv_output_dim(*ih as i64, *kh as i64, stride, padding, dilation) as usize,
                    ),
                    _ => Dimension::Symbolic("H_out".into()),
                };
                let w_out = match (&input[3], &kernel[3]) {
                    (Dimension::Constant(iw), Dimension::Constant(kw)) => Dimension::Constant(
                        conv_output_dim(*iw as i64, *kw as i64, stride, padding, dilation) as usize,
                    ),
                    _ => Dimension::Symbolic("W_out".into()),
                };

                Ok(vec![TensorTypeInfo {
                    shape: vec![n, cout, h_out, w_out],
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Conv1D ──
            TensorOp::Conv1D => {
                if inputs.len() < 2 {
                    return Err("conv1d requires at least 2 inputs".into());
                }
                let input = &inputs[0].shape;
                let kernel = &inputs[1].shape;
                if input.len() != 3 || kernel.len() != 3 {
                    return Err("conv1d: input [N,C,L] and kernel [Cout,Cin,K]".into());
                }
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(attrs, "dilation", 1);
                let n = input[0].clone();
                let cout = kernel[0].clone();
                let l_out = match (&input[2], &kernel[2]) {
                    (Dimension::Constant(il), Dimension::Constant(kl)) => Dimension::Constant(
                        conv_output_dim(*il as i64, *kl as i64, stride, padding, dilation) as usize,
                    ),
                    _ => Dimension::Symbolic("L_out".into()),
                };
                Ok(vec![TensorTypeInfo {
                    shape: vec![n, cout, l_out],
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Conv3D ──
            TensorOp::Conv3D => {
                if inputs.len() < 2 {
                    return Err("conv3d requires at least 2 inputs".into());
                }
                let input = &inputs[0].shape;
                let kernel = &inputs[1].shape;
                if input.len() != 5 || kernel.len() != 5 {
                    return Err("conv3d: input [N,C,D,H,W] and kernel [Cout,Cin,Kd,Kh,Kw]".into());
                }
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(attrs, "dilation", 1);
                let n = input[0].clone();
                let cout = kernel[0].clone();
                let dims: Vec<Dimension> = (2..5)
                    .map(|i| match (&input[i], &kernel[i]) {
                        (Dimension::Constant(iv), Dimension::Constant(kv)) => Dimension::Constant(
                            conv_output_dim(*iv as i64, *kv as i64, stride, padding, dilation)
                                as usize,
                        ),
                        _ => Dimension::Symbolic(format!("dim{}_out", i)),
                    })
                    .collect();
                Ok(vec![TensorTypeInfo {
                    shape: vec![n, cout, dims[0].clone(), dims[1].clone(), dims[2].clone()],
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Pooling ──
            // The second input, if present, is a kernel-shaped tensor whose
            // spatial dims give the pooling window (mirroring how conv reads
            // its window from the kernel tensor) — this matches
            // test_shape_max_pool2d. Default stride is the window size
            // (non-overlapping pooling, the standard framework default when
            // stride is unset), overridable via a `stride` attr; `padding`
            // defaults to 0. With only 1 input (no window given), the
            // pooling window is unknown, so the shape is left unchanged.
            TensorOp::MaxPool2D | TensorOp::AvgPool2D => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                let Some(kernel) = inputs.get(1) else {
                    return Ok(vec![inputs[0].clone()]);
                };
                let input = &inputs[0].shape;
                let kh = kernel.shape.first().and_then(|d| d.static_value());
                let kw = kernel.shape.get(1).and_then(|d| d.static_value());
                if input.len() < 4 {
                    return Err(format!("{}: input must be 4D [N,C,H,W]", op.name()));
                }
                let padding = spatial_param(attrs, "padding", 0);
                let mut out = input.clone();
                if let (Some(kh), Dimension::Constant(ih)) = (kh, &input[2]) {
                    let stride = spatial_param(attrs, "stride", kh as i64);
                    out[2] = Dimension::Constant(conv_output_dim(
                        *ih as i64, kh as i64, stride, padding, 1,
                    ) as usize);
                }
                if let (Some(kw), Dimension::Constant(iw)) = (kw, &input[3]) {
                    let stride = spatial_param(attrs, "stride", kw as i64);
                    out[3] = Dimension::Constant(conv_output_dim(
                        *iw as i64, kw as i64, stride, padding, 1,
                    ) as usize);
                }
                Ok(vec![TensorTypeInfo {
                    shape: out,
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // Adaptive pooling targets a caller-specified output size rather
            // than deriving one from a kernel/stride, and no attrs schema
            // for that output size exists yet in this codebase — simplified
            // to the input shape unchanged, same as MaxPool2D/AvgPool2D
            // without a kernel input, until one is added.
            TensorOp::AdaptiveAvgPool2D => {
                if inputs.is_empty() {
                    return Err("adaptive_avgpool2d requires 1 input".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            TensorOp::GlobalAvgPool => {
                if inputs.is_empty() {
                    return Err("global_avgpool requires 1 input".into());
                }
                let shape = &inputs[0].shape;
                if shape.len() < 3 {
                    return Err("global_avgpool: input must be at least 3D [N,C,...]".into());
                }
                // [N, C, ...] -> [N, C, 1, 1, ...]
                let mut out = vec![shape[0].clone(), shape[1].clone()];
                for _ in 2..shape.len() {
                    out.push(Dimension::Constant(1));
                }
                Ok(vec![TensorTypeInfo {
                    shape: out,
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Attention variants ──
            TensorOp::Attention
            | TensorOp::MultiHeadAttention
            | TensorOp::MultiQueryAttention
            | TensorOp::GroupedQueryAttention
            | TensorOp::FlashAttention
            | TensorOp::SlidingWindowAttention
            | TensorOp::CrossAttention
            | TensorOp::PagedAttention
            | TensorOp::GradAttention => {
                if inputs.len() < 3 {
                    return Err("attention requires at least 3 inputs (Q, K, V)".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── Recurrent ──
            TensorOp::LSTMCell => {
                if inputs.len() < 2 {
                    return Err("lstm_cell requires input and hidden state".into());
                }
                // Returns (h_new, c_new) with same shape as hidden
                Ok(vec![inputs[1].clone(), inputs[1].clone()])
            }

            TensorOp::GRUCell | TensorOp::RNNCell => {
                if inputs.len() < 2 {
                    return Err(format!("{} requires input and hidden state", op.name()));
                }
                Ok(vec![inputs[1].clone()])
            }

            // ── Shape / zero-flop ops ──
            TensorOp::Reshape
            | TensorOp::Transpose
            | TensorOp::Squeeze
            | TensorOp::Unsqueeze
            | TensorOp::Permute
            | TensorOp::Expand
            | TensorOp::Slice
            | TensorOp::Pad
            | TensorOp::Tile => {
                // These need target shape from attributes; passthrough for now
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── Concat ──
            TensorOp::Concat => {
                if inputs.is_empty() {
                    return Err("concat requires at least 1 input".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── TopK / Sort ──
            TensorOp::TopK | TensorOp::Sort => {
                if inputs.is_empty() {
                    return Err(format!("{} requires 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── FFT / IFFT ──
            TensorOp::FFT | TensorOp::IFFT => {
                if inputs.is_empty() {
                    return Err(format!("{} requires 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── SVD: returns U, S, V ──
            TensorOp::SVD => {
                if inputs.is_empty() {
                    return Err("svd requires 1 input".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── Where: condition, x, y -> x ──
            TensorOp::Where | TensorOp::Clamp => {
                if inputs.len() < 2 {
                    return Err(format!("{} requires at least 2 inputs", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            _ => {
                // For ops not yet handled, passthrough first input or empty
                if !inputs.is_empty() {
                    Ok(vec![inputs[0].clone()])
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    pub fn compute_flops(
        op: &TensorOp,
        inputs: &[&TensorTypeInfo],
        attrs: Option<&Attributes>,
    ) -> Option<u64> {
        match op {
            TensorOp::MatMul | TensorOp::SparseMatMul => {
                if inputs.len() != 2 {
                    return None;
                }
                let a = &inputs[0].shape;
                let b = &inputs[1].shape;
                let m = a.get(a.len().checked_sub(2)?)?.static_value()? as u64;
                let k = a.last()?.static_value()? as u64;
                let n = b.last()?.static_value()? as u64;
                let batch: u64 = a[..a.len() - 2]
                    .iter()
                    .filter_map(|d| d.static_value())
                    .map(|v| v as u64)
                    .product::<u64>()
                    .max(1);
                Some(2 * batch * m * n * k)
            }

            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => {
                if inputs.is_empty() {
                    return None;
                }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            TensorOp::ReLU
            | TensorOp::Sigmoid
            | TensorOp::Tanh
            | TensorOp::LeakyReLU
            | TensorOp::ELU
            | TensorOp::HardSigmoid => {
                if inputs.is_empty() {
                    return None;
                }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            TensorOp::GeLU | TensorOp::SiLU | TensorOp::Mish | TensorOp::HardSwish => {
                if inputs.is_empty() {
                    return None;
                }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(8 * n)
            }

            TensorOp::Softmax => {
                if inputs.is_empty() {
                    return None;
                }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::LayerNorm
            | TensorOp::RMSNorm
            | TensorOp::GroupNorm
            | TensorOp::InstanceNorm => {
                if inputs.is_empty() {
                    return None;
                }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(7 * n)
            }

            TensorOp::BatchNorm => {
                if inputs.is_empty() {
                    return None;
                }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::Linear => {
                if inputs.len() < 2 {
                    return None;
                }
                let x = &inputs[0].shape;
                let w = &inputs[1].shape;
                let m: u64 = x[..x.len() - 1]
                    .iter()
                    .filter_map(|d| d.static_value())
                    .map(|v| v as u64)
                    .product::<u64>()
                    .max(1);
                let k = x.last()?.static_value()? as u64;
                let n = w.last()?.static_value()? as u64;
                Some(2 * m * n * k + n)
            }

            TensorOp::Conv2D | TensorOp::DepthwiseConv2D | TensorOp::DilatedConv2D => {
                if inputs.len() < 2 {
                    return None;
                }
                let kernel = &inputs[1].shape;
                let cout = kernel[0].static_value()? as u64;
                let cin = kernel[1].static_value()? as u64;
                let kh = kernel[2].static_value()? as u64;
                let kw = kernel[3].static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input[0].static_value()? as u64;
                let ih = input[2].static_value()? as u64;
                let iw = input[3].static_value()? as u64;
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(
                    attrs,
                    "dilation",
                    if matches!(op, TensorOp::DilatedConv2D) {
                        2
                    } else {
                        1
                    },
                );
                let oh = conv_output_dim(ih as i64, kh as i64, stride, padding, dilation) as u64;
                let ow = conv_output_dim(iw as i64, kw as i64, stride, padding, dilation) as u64;
                Some(2 * n * cout * cin * kh * kw * oh * ow)
            }

            TensorOp::Conv1D => {
                if inputs.len() < 2 {
                    return None;
                }
                let kernel = &inputs[1].shape;
                let cout = kernel[0].static_value()? as u64;
                let cin = kernel[1].static_value()? as u64;
                let k = kernel[2].static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input[0].static_value()? as u64;
                let il = input[2].static_value()? as u64;
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(attrs, "dilation", 1);
                let ol = conv_output_dim(il as i64, k as i64, stride, padding, dilation) as u64;
                Some(2 * n * cout * cin * k * ol)
            }

            TensorOp::Conv3D => {
                if inputs.len() < 2 {
                    return None;
                }
                let kernel = &inputs[1].shape;
                let cout = kernel.first()?.static_value()? as u64;
                let cin = kernel.get(1)?.static_value()? as u64;
                let kd = kernel.get(2)?.static_value()? as u64;
                let kh = kernel.get(3)?.static_value()? as u64;
                let kw = kernel.get(4)?.static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input.first()?.static_value()? as u64;
                let id = input.get(2)?.static_value()? as u64;
                let ih = input.get(3)?.static_value()? as u64;
                let iw = input.get(4)?.static_value()? as u64;
                let stride = spatial_param(attrs, "stride", 1);
                let padding = spatial_param(attrs, "padding", 0);
                let dilation = spatial_param(attrs, "dilation", 1);
                let od = conv_output_dim(id as i64, kd as i64, stride, padding, dilation) as u64;
                let oh = conv_output_dim(ih as i64, kh as i64, stride, padding, dilation) as u64;
                let ow = conv_output_dim(iw as i64, kw as i64, stride, padding, dilation) as u64;
                Some(2 * n * cout * cin * kd * kh * kw * od * oh * ow)
            }

            // Attention variants: 2*B*H*(S^2*D + S*D^2)
            TensorOp::Attention
            | TensorOp::MultiHeadAttention
            | TensorOp::MultiQueryAttention
            | TensorOp::GroupedQueryAttention
            | TensorOp::FlashAttention
            | TensorOp::SlidingWindowAttention
            | TensorOp::CrossAttention => {
                if inputs.is_empty() {
                    return None;
                }
                let shape = &inputs[0].shape;
                if shape.len() < 3 {
                    return None;
                }
                let b = shape[0].static_value().unwrap_or(1) as u64;
                let s = shape[shape.len() - 2].static_value()? as u64;
                let d = shape.last()?.static_value()? as u64;
                let h = if shape.len() >= 4 {
                    shape[1].static_value().unwrap_or(1) as u64
                } else {
                    1
                };
                Some(4 * b * h * s * s * d)
            }

            // Recurrent
            TensorOp::LSTMCell => {
                // 4 * (input_size + hidden_size) * hidden_size * 2
                if inputs.len() < 2 {
                    return None;
                }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(8 * (input_size + hidden_size) * hidden_size)
            }

            TensorOp::GRUCell => {
                if inputs.len() < 2 {
                    return None;
                }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(6 * (input_size + hidden_size) * hidden_size)
            }

            TensorOp::RNNCell => {
                if inputs.len() < 2 {
                    return None;
                }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(2 * (input_size + hidden_size) * hidden_size)
            }

            // FFT: 5*N*log2(N)
            TensorOp::FFT | TensorOp::IFFT => {
                if inputs.is_empty() {
                    return None;
                }
                let n = element_count(&inputs[0].shape)? as u64;
                if n == 0 {
                    return Some(0);
                }
                let log2n = (n as f64).log2().ceil() as u64;
                Some(5 * n * log2n)
            }

            // Pooling
            TensorOp::MaxPool2D
            | TensorOp::AvgPool2D
            | TensorOp::AdaptiveAvgPool2D
            | TensorOp::GlobalAvgPool => {
                if inputs.is_empty() {
                    return None;
                }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            // Zero-flop ops
            _ if op.is_zero_flop() => Some(0),

            _ => None,
        }
    }

    /// Sums every input's bytes plus the output's bytes — memory traffic
    /// includes writing the result, not just reading the operands. Every op
    /// family used to only get this for `MatMul`/`SparseMatMul`; every other
    /// op (Conv*, Attention, Norm, pooling, activations, ...) silently
    /// omitted the output entirely, understating real traffic (e.g. by
    /// ~46% for a typical Conv2D).
    pub fn compute_memory_bytes(
        op: &TensorOp,
        inputs: &[&TensorTypeInfo],
        attrs: Option<&Attributes>,
    ) -> Option<u64> {
        let input_bytes: u64 = inputs
            .iter()
            .filter_map(|i| tensor_bytes(i).map(|b| b as u64))
            .sum();
        let output_bytes: u64 = Self::infer_output_shape(op, inputs, attrs)
            .ok()
            .map(|outs| {
                outs.iter()
                    .filter_map(|o| tensor_info_bytes(o).map(|b| b as u64))
                    .sum()
            })
            .unwrap_or(0);
        Some(input_bytes + output_bytes)
    }
}

fn broadcast_shapes(a: &[Dimension], b: &[Dimension]) -> Result<Vec<Dimension>, String> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let da = if i < a.len() {
            Some(&a[a.len() - 1 - i])
        } else {
            None
        };
        let db = if i < b.len() {
            Some(&b[b.len() - 1 - i])
        } else {
            None
        };

        let dim = match (da, db) {
            (Some(a_dim), Some(b_dim)) => match (a_dim.static_value(), b_dim.static_value()) {
                (Some(a_val), Some(b_val)) => {
                    if a_val == b_val {
                        Dimension::Constant(a_val)
                    } else if a_val == 1 {
                        Dimension::Constant(b_val)
                    } else if b_val == 1 {
                        Dimension::Constant(a_val)
                    } else {
                        return Err(format!("Shape broadcast error: {} vs {}", a_val, b_val));
                    }
                }
                _ => Dimension::Symbolic("broadcast".into()),
            },
            (Some(d), None) | (None, Some(d)) => d.clone(),
            (None, None) => unreachable!(),
        };
        result.push(dim);
    }

    result.reverse();
    Ok(result)
}

fn element_count(shape: &[Dimension]) -> Option<usize> {
    let mut count = 1usize;
    for dim in shape {
        count = count.checked_mul(dim.static_value()?)?;
    }
    Some(count)
}

fn tensor_bytes(info: &TensorTypeInfo) -> Option<usize> {
    Some(element_count(&info.shape)? * info.dtype.byte_size())
}

fn tensor_info_bytes(info: &TensorTypeInfo) -> Option<usize> {
    tensor_bytes(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::attributes::Attribute;
    use lift_core::types::{DataType, MemoryLayout};

    fn make_tensor(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
        TensorTypeInfo {
            shape: shape.into_iter().map(Dimension::Constant).collect(),
            dtype,
            layout: MemoryLayout::Contiguous,
        }
    }

    #[test]
    fn test_matmul_shape() {
        let a = make_tensor(vec![2, 3, 4], DataType::FP32);
        let b = make_tensor(vec![2, 4, 5], DataType::FP32);
        let result =
            ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b], None).unwrap();
        assert_eq!(result.len(), 1);
        let shape = &result[0].shape;
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0].static_value(), Some(2));
        assert_eq!(shape[1].static_value(), Some(3));
        assert_eq!(shape[2].static_value(), Some(5));
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = make_tensor(vec![3, 4], DataType::FP32);
        let b = make_tensor(vec![5, 6], DataType::FP32);
        let result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_flops() {
        let a = make_tensor(vec![2, 3], DataType::FP32);
        let b = make_tensor(vec![3, 4], DataType::FP32);
        let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b], None);
        assert_eq!(flops, Some(2 * 2 * 4 * 3)); // 2*M*N*K
    }

    #[test]
    fn test_relu_shape() {
        let a = make_tensor(vec![2, 3, 4], DataType::FP32);
        let result = ShapeInference::infer_output_shape(&TensorOp::ReLU, &[&a], None).unwrap();
        assert_eq!(result[0].shape, a.shape);
    }

    #[test]
    fn test_linear_shape() {
        let x = make_tensor(vec![1, 784], DataType::FP32);
        let w = make_tensor(vec![784, 64], DataType::FP32);
        let b = make_tensor(vec![64], DataType::FP32);
        let result =
            ShapeInference::infer_output_shape(&TensorOp::Linear, &[&x, &w, &b], None).unwrap();
        assert_eq!(result[0].shape[0].static_value(), Some(1));
        assert_eq!(result[0].shape[1].static_value(), Some(64));
    }

    #[test]
    fn test_conv2d_shape() {
        let input = make_tensor(vec![1, 3, 28, 28], DataType::FP32);
        let kernel = make_tensor(vec![16, 3, 5, 5], DataType::FP32);
        let result =
            ShapeInference::infer_output_shape(&TensorOp::Conv2D, &[&input, &kernel], None)
                .unwrap();
        assert_eq!(result[0].shape[0].static_value(), Some(1));
        assert_eq!(result[0].shape[1].static_value(), Some(16));
        assert_eq!(result[0].shape[2].static_value(), Some(24)); // 28-5+1
        assert_eq!(result[0].shape[3].static_value(), Some(24));
    }

    /// Regression test: Conv2D used to ignore stride/padding/dilation
    /// entirely (no attrs were even passed in), always computing
    /// `in - kernel + 1` regardless of what the op actually specified.
    #[test]
    fn test_conv2d_shape_honours_stride_padding_dilation() {
        let input = make_tensor(vec![1, 3, 28, 28], DataType::FP32);
        let kernel = make_tensor(vec![16, 3, 3, 3], DataType::FP32);

        // stride=2, padding=1, dilation=1: out = floor((28+2-2-1)/2)+1 = 14
        let mut attrs = Attributes::new();
        attrs.set("stride", Attribute::Integer(2));
        attrs.set("padding", Attribute::Integer(1));
        let result =
            ShapeInference::infer_output_shape(&TensorOp::Conv2D, &[&input, &kernel], Some(&attrs))
                .unwrap();
        assert_eq!(result[0].shape[2].static_value(), Some(14));
        assert_eq!(result[0].shape[3].static_value(), Some(14));

        let flops =
            ShapeInference::compute_flops(&TensorOp::Conv2D, &[&input, &kernel], Some(&attrs))
                .unwrap();
        assert_eq!(flops, 2 * 16 * 3 * 3 * 3 * 14 * 14);
    }

    /// Regression test: DilatedConv2D used to compute the exact same shape
    /// as a plain Conv2D, silently ignoring dilation. With no attrs given it
    /// now defaults to dilation=2 (the point of the op), producing a
    /// different, smaller output than Conv2D would for the same input.
    #[test]
    fn test_dilated_conv2d_differs_from_plain_conv2d_by_default() {
        let input = make_tensor(vec![1, 3, 28, 28], DataType::FP32);
        let kernel = make_tensor(vec![16, 3, 3, 3], DataType::FP32);

        let plain = ShapeInference::infer_output_shape(&TensorOp::Conv2D, &[&input, &kernel], None)
            .unwrap();
        let dilated =
            ShapeInference::infer_output_shape(&TensorOp::DilatedConv2D, &[&input, &kernel], None)
                .unwrap();

        assert_eq!(plain[0].shape[2].static_value(), Some(26)); // 28-3+1
                                                                // dilation=2: out = floor((28 - 2*(3-1) - 1)/1)+1 = 24
        assert_eq!(dilated[0].shape[2].static_value(), Some(24));
        assert_ne!(
            plain[0].shape[2], dilated[0].shape[2],
            "DilatedConv2D must not silently behave like Conv2D"
        );
    }

    /// Regression test: MaxPool2D/AvgPool2D used to return the input shape
    /// unchanged ("simplified"), never reducing spatial dims at all.
    #[test]
    fn test_maxpool2d_reduces_spatial_dims() {
        let input = make_tensor(vec![1, 64, 32, 32], DataType::FP32);
        let kernel = make_tensor(vec![2, 2], DataType::FP32);
        let result =
            ShapeInference::infer_output_shape(&TensorOp::MaxPool2D, &[&input, &kernel], None)
                .unwrap();
        assert_eq!(result[0].shape[0].static_value(), Some(1));
        assert_eq!(result[0].shape[1].static_value(), Some(64));
        // Default stride = kernel size (non-overlapping): 32/2 = 16.
        assert_eq!(result[0].shape[2].static_value(), Some(16));
        assert_eq!(result[0].shape[3].static_value(), Some(16));
    }

    /// Regression test: compute_memory_bytes used to add the output's bytes
    /// only for MatMul/SparseMatMul — every other op (Conv2D here) silently
    /// omitted the output from the memory-traffic total.
    #[test]
    fn test_compute_memory_bytes_includes_output_for_every_op() {
        let input = make_tensor(vec![1, 3, 28, 28], DataType::FP32); // 2352 elems
        let kernel = make_tensor(vec![16, 3, 5, 5], DataType::FP32); // 1200 elems
        let mem = ShapeInference::compute_memory_bytes(&TensorOp::Conv2D, &[&input, &kernel], None)
            .unwrap();
        // input + kernel + output(1,16,24,24) elements, all FP32 (4 bytes/elem).
        let expected = (2352 + 1200 + 16 * 24 * 24) * 4;
        assert_eq!(mem, expected as u64);
    }
}
