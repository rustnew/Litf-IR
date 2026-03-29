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

            TensorOp::Neg | TensorOp::ReLU | TensorOp::GeLU | TensorOp::SiLU |
            TensorOp::Sigmoid | TensorOp::Tanh => {
                if inputs.is_empty() {
                    return Err(format!("{} requires at least 1 input", op.name()));
                }
                Ok(vec![inputs[0].clone()])
            }

            TensorOp::Softmax => {
                if inputs.is_empty() {
                    return Err("softmax requires 1 input".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            TensorOp::MatMul => {
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

                // Check inner dimensions match
                let k_a = &a[a.len() - 1];
                let k_b = &b[b.len() - 2];
                if let (Some(ka), Some(kb)) = (k_a.static_value(), k_b.static_value()) {
                    if ka != kb {
                        return Err(format!(
                            "matmul inner dimension mismatch: {} vs {}", ka, kb
                        ));
                    }
                }

                // Batch dimensions
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

            TensorOp::Linear => {
                if inputs.len() < 2 {
                    return Err("linear requires at least 2 inputs (x, W)".into());
                }
                let x = &inputs[0].shape;
                let w = &inputs[1].shape;
                if x.len() < 1 || w.len() != 2 {
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

            TensorOp::LayerNorm | TensorOp::RMSNorm => {
                if inputs.is_empty() {
                    return Err("layernorm requires at least 1 input".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            TensorOp::Conv2D => {
                if inputs.len() < 2 {
                    return Err("conv2d requires at least 2 inputs (input, kernel)".into());
                }
                let input = &inputs[0].shape;
                let kernel = &inputs[1].shape;
                if input.len() != 4 || kernel.len() != 4 {
                    return Err("conv2d: input and kernel must be 4D (NCHW)".into());
                }

                // Simplified: assume stride=1, padding=0
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

            TensorOp::Attention => {
                // Attention: Q, K, V -> output with same shape as Q
                if inputs.len() < 3 {
                    return Err("attention requires at least 3 inputs (Q, K, V)".into());
                }
                Ok(vec![inputs[0].clone()])
            }

            _ => {
                // For ops not yet handled, return empty (caller must handle)
                Ok(Vec::new())
            }
        }
    }

    pub fn compute_flops(op: &TensorOp, inputs: &[&TensorTypeInfo]) -> Option<u64> {
        match op {
            TensorOp::MatMul => {
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
                let n = element_count(&inputs[0].shape)? as u64;
                Some(n)
            }

            TensorOp::ReLU | TensorOp::Sigmoid | TensorOp::Tanh => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(n)
            }

            TensorOp::GeLU | TensorOp::SiLU => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::Softmax => {
                if inputs.is_empty() { return None; }
                let n = element_count(&inputs[0].shape)? as u64;
                Some(5 * n)
            }

            TensorOp::LayerNorm | TensorOp::RMSNorm => {
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
                Some(2 * m * n * k + n) // + bias
            }

            TensorOp::Conv2D => {
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
                let oh = ih - kh + 1;
                let ow = iw - kw + 1;
                Some(2 * n * cout * cin * kh * kw * oh * ow)
            }

            TensorOp::Attention => {
                // Standard attention: 4*B*H*S*S*D
                if inputs.is_empty() { return None; }
                let shape = &inputs[0].shape;
                if shape.len() < 3 { return None; }
                let b = shape[0].static_value().unwrap_or(1) as u64;
                let s = shape[shape.len() - 2].static_value()? as u64;
                let d = shape.last()?.static_value()? as u64;
                let h = if shape.len() >= 4 { shape[1].static_value().unwrap_or(1) as u64 } else { 1 };
                Some(4 * b * h * s * s * d)
            }

            TensorOp::Reshape | TensorOp::Transpose => Some(0),

            _ => None,
        }
    }

    pub fn compute_memory_bytes(op: &TensorOp, inputs: &[&TensorTypeInfo]) -> Option<u64> {
        match op {
            TensorOp::MatMul => {
                if inputs.len() != 2 { return None; }
                let a_bytes = tensor_bytes(&inputs[0])? as u64;
                let b_bytes = tensor_bytes(&inputs[1])? as u64;
                // Output size
                let out_shape = Self::infer_output_shape(op, inputs).ok()?;
                let out_bytes = if let Some(out) = out_shape.first() {
                    tensor_info_bytes(out)? as u64
                } else { 0 };
                Some(a_bytes + b_bytes + out_bytes)
            }
            _ => {
                // Generic: sum of all input sizes
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
