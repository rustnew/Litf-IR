use lift_core::types::{Dimension, TensorTypeInfo};
use crate::ops::TensorOp;

#[derive(Debug)]
pub struct ShapeInference;

impl ShapeInference {
    pub fn infer_output_shape(
        op: &TensorOp,
        inputs: &[&TensorTypeInfo],
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
            TensorOp::Neg | TensorOp::ReLU | TensorOp::GeLU | TensorOp::SiLU |
            TensorOp::Sigmoid | TensorOp::Tanh |
            TensorOp::LeakyReLU | TensorOp::ELU | TensorOp::Mish |
            TensorOp::HardSwish | TensorOp::HardSigmoid |
            TensorOp::Softmax | TensorOp::Cumsum |
            TensorOp::Quantize | TensorOp::Dequantize |
            TensorOp::QuantizeInt4 | TensorOp::DequantizeInt4 |
            TensorOp::QuantizeFp8 | TensorOp::DequantizeFp8 |
            TensorOp::Checkpoint | TensorOp::Offload |
            TensorOp::GradReLU | TensorOp::GradGeLU | TensorOp::GradSoftmax => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            // ── Normalisation (shape-preserving) ──
            TensorOp::LayerNorm | TensorOp::RMSNorm | TensorOp::BatchNorm |
            TensorOp::GroupNorm | TensorOp::InstanceNorm |
            TensorOp::GradLayerNorm => {
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
                        return Err(format!(
                            "matmul inner dimension mismatch: {} vs {}", ka, kb
                        ));
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

                let n = input[0].clone();
                let cout = kernel[0].clone();
                let h_out = match (&input[2], &kernel[2]) {
                    (Dimension::Constant(ih), Dimension::Constant(kh)) => {
                        Dimension::Constant(ih - kh + 1)
                    }
                    _ => Dimension::Symbolic("H_out".into()),
                };
                let w_out = match (&input[3], &kernel[3]) {
                    (Dimension::Constant(iw), Dimension::Constant(kw)) => {
                        Dimension::Constant(iw - kw + 1)
                    }
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
                let n = input[0].clone();
                let cout = kernel[0].clone();
                let l_out = match (&input[2], &kernel[2]) {
                    (Dimension::Constant(il), Dimension::Constant(kl)) => {
                        Dimension::Constant(il - kl + 1)
                    }
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
                let n = input[0].clone();
                let cout = kernel[0].clone();
                let dims: Vec<Dimension> = (2..5).map(|i| {
                    match (&input[i], &kernel[i]) {
                        (Dimension::Constant(iv), Dimension::Constant(kv)) => {
                            Dimension::Constant(iv - kv + 1)
                        }
                        _ => Dimension::Symbolic(format!("dim{}_out", i)),
                    }
                }).collect();
                Ok(vec![TensorTypeInfo {
                    shape: vec![n, cout, dims[0].clone(), dims[1].clone(), dims[2].clone()],
                    dtype: inputs[0].dtype,
                    layout: inputs[0].layout,
                }])
            }

            // ── Pooling ──
            TensorOp::MaxPool2D | TensorOp::AvgPool2D => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                // Simplified: returns same shape (caller should use attrs for kernel/stride)
                Ok(vec![inputs[0].clone()])
            }

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
            TensorOp::Attention | TensorOp::MultiHeadAttention |
            TensorOp::MultiQueryAttention | TensorOp::GroupedQueryAttention |
            TensorOp::FlashAttention | TensorOp::SlidingWindowAttention |
            TensorOp::CrossAttention | TensorOp::PagedAttention |
            TensorOp::GradAttention => {
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
            TensorOp::Reshape | TensorOp::Transpose | TensorOp::Squeeze |
            TensorOp::Unsqueeze | TensorOp::Permute | TensorOp::Expand |
            TensorOp::Slice | TensorOp::Pad | TensorOp::Tile => {
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

    pub fn compute_flops(op: &TensorOp, inputs: &[&TensorTypeInfo]) -> Option<u64> {
        match op {
            TensorOp::MatMul | TensorOp::SparseMatMul => {
                if inputs.len() != 2 { return None; }
                let a = &inputs[0].shape;
                let b = &inputs[1].shape;
                let m = a.get(a.len().checked_sub(2)?)?.static_value()? as u64;
                let k = a.last()?.static_value()? as u64;
                let n = b.last()?.static_value()? as u64;
                let batch: u64 = a[..a.len() - 2].iter()
                    .filter_map(|d| d.static_value())
                    .map(|v| v as u64)
                    .product::<u64>()
                    .max(1);
                Some(2 * batch * m * n * k)
            }

            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => {
                if inputs.is_empty() { return None; }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            TensorOp::ReLU | TensorOp::Sigmoid | TensorOp::Tanh |
            TensorOp::LeakyReLU | TensorOp::ELU | TensorOp::HardSigmoid => {
                if inputs.is_empty() { return None; }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            TensorOp::GeLU | TensorOp::SiLU | TensorOp::Mish | TensorOp::HardSwish => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(8 * n)
            }

            TensorOp::Softmax => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::LayerNorm | TensorOp::RMSNorm |
            TensorOp::GroupNorm | TensorOp::InstanceNorm => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(7 * n)
            }

            TensorOp::BatchNorm => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::Linear => {
                if inputs.len() < 2 { return None; }
                let x = &inputs[0].shape;
                let w = &inputs[1].shape;
                let m: u64 = x[..x.len() - 1].iter()
                    .filter_map(|d| d.static_value())
                    .map(|v| v as u64)
                    .product::<u64>()
                    .max(1);
                let k = x.last()?.static_value()? as u64;
                let n = w.last()?.static_value()? as u64;
                Some(2 * m * n * k + n)
            }

            TensorOp::Conv2D | TensorOp::DepthwiseConv2D | TensorOp::DilatedConv2D => {
                if inputs.len() < 2 { return None; }
                let kernel = &inputs[1].shape;
                let cout = kernel[0].static_value()? as u64;
                let cin = kernel[1].static_value()? as u64;
                let kh = kernel[2].static_value()? as u64;
                let kw = kernel[3].static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input[0].static_value()? as u64;
                let ih = input[2].static_value()? as u64;
                let iw = input[3].static_value()? as u64;
                let oh = ih.saturating_sub(kh) + 1;
                let ow = iw.saturating_sub(kw) + 1;
                Some(2 * n * cout * cin * kh * kw * oh * ow)
            }

            TensorOp::Conv1D => {
                if inputs.len() < 2 { return None; }
                let kernel = &inputs[1].shape;
                let cout = kernel[0].static_value()? as u64;
                let cin = kernel[1].static_value()? as u64;
                let k = kernel[2].static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input[0].static_value()? as u64;
                let il = input[2].static_value()? as u64;
                let ol = il.saturating_sub(k) + 1;
                Some(2 * n * cout * cin * k * ol)
            }

            TensorOp::Conv3D => {
                if inputs.len() < 2 { return None; }
                let kernel = &inputs[1].shape;
                let cout = kernel.get(0)?.static_value()? as u64;
                let cin = kernel.get(1)?.static_value()? as u64;
                let kd = kernel.get(2)?.static_value()? as u64;
                let kh = kernel.get(3)?.static_value()? as u64;
                let kw = kernel.get(4)?.static_value()? as u64;
                let input = &inputs[0].shape;
                let n = input.get(0)?.static_value()? as u64;
                let id = input.get(2)?.static_value()? as u64;
                let ih = input.get(3)?.static_value()? as u64;
                let iw = input.get(4)?.static_value()? as u64;
                let od = id.saturating_sub(kd) + 1;
                let oh = ih.saturating_sub(kh) + 1;
                let ow = iw.saturating_sub(kw) + 1;
                Some(2 * n * cout * cin * kd * kh * kw * od * oh * ow)
            }

            // Attention variants: 2*B*H*(S^2*D + S*D^2)
            TensorOp::Attention | TensorOp::MultiHeadAttention |
            TensorOp::MultiQueryAttention | TensorOp::GroupedQueryAttention |
            TensorOp::FlashAttention | TensorOp::SlidingWindowAttention |
            TensorOp::CrossAttention => {
                if inputs.is_empty() { return None; }
                let shape = &inputs[0].shape;
                if shape.len() < 3 { return None; }
                let b = shape[0].static_value().unwrap_or(1) as u64;
                let s = shape[shape.len() - 2].static_value()? as u64;
                let d = shape.last()?.static_value()? as u64;
                let h = if shape.len() >= 4 {
                    shape[1].static_value().unwrap_or(1) as u64
                } else { 1 };
                Some(4 * b * h * s * s * d)
            }

            // Recurrent
            TensorOp::LSTMCell => {
                // 4 * (input_size + hidden_size) * hidden_size * 2
                if inputs.len() < 2 { return None; }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(8 * (input_size + hidden_size) * hidden_size)
            }

            TensorOp::GRUCell => {
                if inputs.len() < 2 { return None; }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(6 * (input_size + hidden_size) * hidden_size)
            }

            TensorOp::RNNCell => {
                if inputs.len() < 2 { return None; }
                let input_size = inputs[0].shape.last()?.static_value()? as u64;
                let hidden_size = inputs[1].shape.last()?.static_value()? as u64;
                Some(2 * (input_size + hidden_size) * hidden_size)
            }

            // FFT: 5*N*log2(N)
            TensorOp::FFT | TensorOp::IFFT => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                if n == 0 { return Some(0); }
                let log2n = (n as f64).log2().ceil() as u64;
                Some(5 * n * log2n)
            }

            // Pooling
            TensorOp::MaxPool2D | TensorOp::AvgPool2D |
            TensorOp::AdaptiveAvgPool2D | TensorOp::GlobalAvgPool => {
                if inputs.is_empty() { return None; }
                Some(element_count(&inputs[0].shape)? as u64)
            }

            // Zero-flop ops
            _ if op.is_zero_flop() => Some(0),

            _ => None,
        }
    }

    pub fn compute_memory_bytes(op: &TensorOp, inputs: &[&TensorTypeInfo]) -> Option<u64> {
        match op {
            TensorOp::MatMul | TensorOp::SparseMatMul => {
                if inputs.len() != 2 { return None; }
                let a_bytes = tensor_bytes(inputs[0])? as u64;
                let b_bytes = tensor_bytes(inputs[1])? as u64;
                let out_shape = Self::infer_output_shape(op, inputs).ok()?;
                let out_bytes = if let Some(out) = out_shape.first() {
                    tensor_info_bytes(out)? as u64
                } else { 0 };
                Some(a_bytes + b_bytes + out_bytes)
            }
            _ => {
                let total: u64 = inputs.iter()
                    .filter_map(|i| tensor_bytes(i).map(|b| b as u64))
                    .sum();
                Some(total)
            }
        }
    }
}

fn broadcast_shapes(a: &[Dimension], b: &[Dimension]) -> Result<Vec<Dimension>, String> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let da = if i < a.len() { Some(&a[a.len() - 1 - i]) } else { None };
        let db = if i < b.len() { Some(&b[b.len() - 1 - i]) } else { None };

        let dim = match (da, db) {
            (Some(a_dim), Some(b_dim)) => {
                match (a_dim.static_value(), b_dim.static_value()) {
                    (Some(a_val), Some(b_val)) => {
                        if a_val == b_val { Dimension::Constant(a_val) }
                        else if a_val == 1 { Dimension::Constant(b_val) }
                        else if b_val == 1 { Dimension::Constant(a_val) }
                        else { return Err(format!(
                            "Shape broadcast error: {} vs {}", a_val, b_val
                        )); }
                    }
                    _ => Dimension::Symbolic("broadcast".into()),
                }
            }
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
        let result = ShapeInference::infer_output_shape(
            &TensorOp::MatMul, &[&a, &b]
        ).unwrap();
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
        let result = ShapeInference::infer_output_shape(
            &TensorOp::MatMul, &[&a, &b]
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_flops() {
        let a = make_tensor(vec![2, 3], DataType::FP32);
        let b = make_tensor(vec![3, 4], DataType::FP32);
        let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
        assert_eq!(flops, Some(2 * 2 * 4 * 3)); // 2*M*N*K
    }

    #[test]
    fn test_relu_shape() {
        let a = make_tensor(vec![2, 3, 4], DataType::FP32);
        let result = ShapeInference::infer_output_shape(
            &TensorOp::ReLU, &[&a]
        ).unwrap();
        assert_eq!(result[0].shape, a.shape);
    }

    #[test]
    fn test_linear_shape() {
        let x = make_tensor(vec![1, 784], DataType::FP32);
        let w = make_tensor(vec![784, 64], DataType::FP32);
        let b = make_tensor(vec![64], DataType::FP32);
        let result = ShapeInference::infer_output_shape(
            &TensorOp::Linear, &[&x, &w, &b]
        ).unwrap();
        assert_eq!(result[0].shape[0].static_value(), Some(1));
        assert_eq!(result[0].shape[1].static_value(), Some(64));
    }

    #[test]
    fn test_conv2d_shape() {
        let input = make_tensor(vec![1, 3, 28, 28], DataType::FP32);
        let kernel = make_tensor(vec![16, 3, 5, 5], DataType::FP32);
        let result = ShapeInference::infer_output_shape(
            &TensorOp::Conv2D, &[&input, &kernel]
        ).unwrap();
        assert_eq!(result[0].shape[0].static_value(), Some(1));
        assert_eq!(result[0].shape[1].static_value(), Some(16));
        assert_eq!(result[0].shape[2].static_value(), Some(24)); // 28-5+1
        assert_eq!(result[0].shape[3].static_value(), Some(24));
    }
}
