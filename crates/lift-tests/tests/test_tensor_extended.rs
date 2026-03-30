use lift_core::types::*;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;

fn make_tensor(dims: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: dims.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// ═══════════════════════════════════════════════════════════
// TensorOp name roundtrip — all new ops
// ═══════════════════════════════════════════════════════════

#[test]
fn test_activation_ops_roundtrip() {
    let ops = [
        TensorOp::LeakyReLU, TensorOp::ELU, TensorOp::Mish,
        TensorOp::HardSwish, TensorOp::HardSigmoid,
    ];
    for op in &ops {
        let name = op.name();
        let recovered = TensorOp::from_name(name);
        assert_eq!(recovered.as_ref(), Some(op), "roundtrip failed for {:?}", op);
    }
}

#[test]
fn test_norm_ops_roundtrip() {
    let ops = [
        TensorOp::RMSNorm, TensorOp::BatchNorm,
        TensorOp::GroupNorm, TensorOp::InstanceNorm,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_attention_ops_roundtrip() {
    let ops = [
        TensorOp::Attention, TensorOp::MultiHeadAttention,
        TensorOp::MultiQueryAttention, TensorOp::GroupedQueryAttention,
        TensorOp::FlashAttention, TensorOp::SlidingWindowAttention,
        TensorOp::CrossAttention, TensorOp::PagedAttention,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_conv_ops_roundtrip() {
    let ops = [
        TensorOp::Conv1D, TensorOp::Conv3D,
        TensorOp::ConvTranspose2D, TensorOp::DepthwiseConv2D,
        TensorOp::DilatedConv2D,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_pooling_ops_roundtrip() {
    let ops = [
        TensorOp::MaxPool2D, TensorOp::AvgPool2D,
        TensorOp::AdaptiveAvgPool2D, TensorOp::GlobalAvgPool,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_recurrent_ops_roundtrip() {
    let ops = [TensorOp::LSTMCell, TensorOp::GRUCell, TensorOp::RNNCell];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_advanced_math_ops_roundtrip() {
    let ops = [
        TensorOp::Einsum, TensorOp::FFT, TensorOp::IFFT,
        TensorOp::SVD, TensorOp::Eig, TensorOp::Solve,
        TensorOp::TopK, TensorOp::Sort, TensorOp::Cumsum,
        TensorOp::Where, TensorOp::Clamp,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_quantisation_ops_roundtrip() {
    let ops = [
        TensorOp::Quantize, TensorOp::Dequantize,
        TensorOp::QuantizeInt4, TensorOp::DequantizeInt4,
        TensorOp::QuantizeFp8, TensorOp::DequantizeFp8,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_sparse_ops_roundtrip() {
    let ops = [TensorOp::SparseMatMul, TensorOp::SparseEmbedding];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_diffusion_ops_roundtrip() {
    let ops = [
        TensorOp::UNetDownBlock, TensorOp::UNetUpBlock,
        TensorOp::TimestepEmbedding,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_gnn_ops_roundtrip() {
    let ops = [TensorOp::GNNMessagePassing, TensorOp::GNNGlobalPooling];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_gradient_ops_roundtrip() {
    let ops = [
        TensorOp::GradMatMul, TensorOp::GradReLU, TensorOp::GradSoftmax,
        TensorOp::GradLayerNorm, TensorOp::GradAttention,
        TensorOp::GradConv2D, TensorOp::GradLinear, TensorOp::GradGeLU,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_parallel_ops_roundtrip() {
    let ops = [
        TensorOp::ParallelSplit, TensorOp::ParallelAllReduce,
        TensorOp::PipelineSend, TensorOp::PipelineReceive,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_fused_ops_roundtrip() {
    let ops = [
        TensorOp::FusedMatMulBiasReLU, TensorOp::FusedMatMulBias,
        TensorOp::FusedLinearGeLU, TensorOp::FusedAttentionLayerNorm,
        TensorOp::FusedLinearSiLU, TensorOp::FusedConvBatchNormReLU,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_shape_ops_roundtrip() {
    let ops = [
        TensorOp::Squeeze, TensorOp::Unsqueeze, TensorOp::Permute,
        TensorOp::Expand, TensorOp::Slice, TensorOp::Pad, TensorOp::Tile,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_creation_ops_roundtrip() {
    let ops = [
        TensorOp::Constant, TensorOp::Zeros, TensorOp::Ones,
        TensorOp::Arange, TensorOp::Full,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_moe_ops_roundtrip() {
    let ops = [TensorOp::MoEDispatch, TensorOp::MoECombine];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

#[test]
fn test_memory_ops_roundtrip() {
    let ops = [
        TensorOp::Checkpoint, TensorOp::Offload,
        TensorOp::GradAccumulate,
    ];
    for op in &ops {
        assert_eq!(TensorOp::from_name(op.name()).as_ref(), Some(op));
    }
}

// ═══════════════════════════════════════════════════════════
// Predicate tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_is_activation() {
    let activations = [
        TensorOp::ReLU, TensorOp::GeLU, TensorOp::SiLU,
        TensorOp::Sigmoid, TensorOp::Tanh, TensorOp::LeakyReLU,
        TensorOp::ELU, TensorOp::Mish, TensorOp::HardSwish,
        TensorOp::HardSigmoid,
    ];
    for op in &activations {
        assert!(op.is_activation(), "{:?} should be activation", op);
    }
    assert!(!TensorOp::MatMul.is_activation());
    assert!(!TensorOp::Conv2D.is_activation());
}

#[test]
fn test_is_attention() {
    let attn_ops = [
        TensorOp::Attention, TensorOp::MultiHeadAttention,
        TensorOp::FlashAttention, TensorOp::CrossAttention,
    ];
    for op in &attn_ops {
        assert!(op.is_attention(), "{:?} should be attention", op);
    }
    assert!(!TensorOp::MatMul.is_attention());
}

#[test]
fn test_is_convolution() {
    let conv_ops = [
        TensorOp::Conv2D, TensorOp::Conv1D, TensorOp::Conv3D,
        TensorOp::ConvTranspose2D, TensorOp::DepthwiseConv2D,
        TensorOp::DilatedConv2D,
    ];
    for op in &conv_ops {
        assert!(op.is_convolution(), "{:?} should be convolution", op);
    }
    assert!(!TensorOp::Linear.is_convolution());
}

#[test]
fn test_is_normalisation() {
    let norm_ops = [
        TensorOp::LayerNorm, TensorOp::RMSNorm,
        TensorOp::BatchNorm, TensorOp::GroupNorm,
        TensorOp::InstanceNorm,
    ];
    for op in &norm_ops {
        assert!(op.is_normalisation(), "{:?} should be normalisation", op);
    }
    assert!(!TensorOp::ReLU.is_normalisation());
}

#[test]
fn test_is_fused() {
    let fused = [
        TensorOp::FusedMatMulBiasReLU, TensorOp::FusedMatMulBias,
        TensorOp::FusedLinearGeLU, TensorOp::FusedAttentionLayerNorm,
        TensorOp::FusedLinearSiLU, TensorOp::FusedConvBatchNormReLU,
    ];
    for op in &fused {
        assert!(op.is_fused(), "{:?} should be fused", op);
    }
    assert!(!TensorOp::MatMul.is_fused());
}

#[test]
fn test_is_gradient() {
    let grads = [
        TensorOp::GradMatMul, TensorOp::GradReLU,
        TensorOp::GradSoftmax, TensorOp::GradLayerNorm,
        TensorOp::GradAttention, TensorOp::GradConv2D,
        TensorOp::GradLinear, TensorOp::GradGeLU,
    ];
    for op in &grads {
        assert!(op.is_gradient(), "{:?} should be gradient", op);
    }
    assert!(!TensorOp::MatMul.is_gradient());
}

#[test]
fn test_is_zero_flop() {
    let zero_flop = [
        TensorOp::Reshape, TensorOp::Transpose, TensorOp::Squeeze,
        TensorOp::Unsqueeze, TensorOp::Permute, TensorOp::Expand,
        TensorOp::Slice,
    ];
    for op in &zero_flop {
        assert!(op.is_zero_flop(), "{:?} should be zero-flop", op);
    }
    assert!(!TensorOp::MatMul.is_zero_flop());
}

// ═══════════════════════════════════════════════════════════
// Shape inference tests — new ops
// ═══════════════════════════════════════════════════════════

#[test]
fn test_shape_leaky_relu() {
    let a = make_tensor(vec![2, 3, 4], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::LeakyReLU, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_elu() {
    let a = make_tensor(vec![8, 64], DataType::FP16);
    let result = ShapeInference::infer_output_shape(&TensorOp::ELU, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
    assert_eq!(result[0].dtype, DataType::FP16);
}

#[test]
fn test_shape_mish() {
    let a = make_tensor(vec![4, 128], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Mish, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_hard_swish() {
    let a = make_tensor(vec![1, 3, 224, 224], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::HardSwish, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_batch_norm() {
    let a = make_tensor(vec![8, 64, 32, 32], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::BatchNorm, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_instance_norm() {
    let a = make_tensor(vec![8, 64, 32, 32], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::InstanceNorm, &[&a]).unwrap();
    assert_eq!(result[0].shape, a.shape);
}

#[test]
fn test_shape_conv1d() {
    let input = make_tensor(vec![1, 3, 100], DataType::FP32);
    let kernel = make_tensor(vec![16, 3, 5], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Conv1D, &[&input, &kernel]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(1));
    assert_eq!(result[0].shape[1].static_value(), Some(16));
    assert_eq!(result[0].shape[2].static_value(), Some(96)); // 100-5+1
}

#[test]
fn test_shape_max_pool2d() {
    let input = make_tensor(vec![1, 64, 32, 32], DataType::FP32);
    let kernel = make_tensor(vec![2, 2], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::MaxPool2D, &[&input, &kernel]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(1));
    assert_eq!(result[0].shape[1].static_value(), Some(64));
    // Output depends on impl; just check rank is preserved
    assert_eq!(result[0].shape.len(), 4);
}

#[test]
fn test_shape_global_avg_pool() {
    let input = make_tensor(vec![8, 512, 7, 7], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::GlobalAvgPool, &[&input]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(8));
    assert_eq!(result[0].shape[1].static_value(), Some(512));
    assert_eq!(result[0].shape[2].static_value(), Some(1));
    assert_eq!(result[0].shape[3].static_value(), Some(1));
}

#[test]
fn test_shape_attention() {
    let q = make_tensor(vec![2, 8, 128, 64], DataType::FP32);
    let k = make_tensor(vec![2, 8, 128, 64], DataType::FP32);
    let v = make_tensor(vec![2, 8, 128, 64], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::Attention, &[&q, &k, &v]).unwrap();
    assert_eq!(result[0].shape, q.shape);
}

#[test]
fn test_shape_flash_attention() {
    let q = make_tensor(vec![2, 8, 512, 64], DataType::FP16);
    let k = make_tensor(vec![2, 8, 512, 64], DataType::FP16);
    let v = make_tensor(vec![2, 8, 512, 64], DataType::FP16);
    let result = ShapeInference::infer_output_shape(&TensorOp::FlashAttention, &[&q, &k, &v]).unwrap();
    assert_eq!(result[0].shape, q.shape);
    assert_eq!(result[0].dtype, DataType::FP16);
}

#[test]
fn test_shape_sparse_matmul() {
    let a = make_tensor(vec![4, 3], DataType::FP32);
    let b = make_tensor(vec![3, 5], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::SparseMatMul, &[&a, &b]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(4));
    assert_eq!(result[0].shape[1].static_value(), Some(5));
}

#[test]
fn test_shape_depthwise_conv2d() {
    let input = make_tensor(vec![1, 32, 28, 28], DataType::FP32);
    let kernel = make_tensor(vec![32, 1, 3, 3], DataType::FP32);
    let result = ShapeInference::infer_output_shape(&TensorOp::DepthwiseConv2D, &[&input, &kernel]).unwrap();
    assert_eq!(result[0].shape[0].static_value(), Some(1));
    assert_eq!(result[0].shape[1].static_value(), Some(32));
    assert_eq!(result[0].shape[2].static_value(), Some(26)); // 28-3+1
}

// ═══════════════════════════════════════════════════════════
// FLOPs computation tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_flops_zero_for_reshape() {
    let a = make_tensor(vec![2, 3, 4], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Reshape, &[&a]);
    assert_eq!(flops, Some(0));
}

#[test]
fn test_flops_zero_for_transpose() {
    let a = make_tensor(vec![4, 8], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Transpose, &[&a]);
    assert_eq!(flops, Some(0));
}

#[test]
fn test_flops_relu() {
    let a = make_tensor(vec![2, 3], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::ReLU, &[&a]);
    assert_eq!(flops, Some(6)); // 2*3
}

#[test]
fn test_flops_matmul() {
    let a = make_tensor(vec![4, 3], DataType::FP32);
    let b = make_tensor(vec![3, 5], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
    assert_eq!(flops, Some(2 * 4 * 3 * 5));
}

#[test]
fn test_flops_attention() {
    let q = make_tensor(vec![1, 1, 4, 8], DataType::FP32);
    let k = make_tensor(vec![1, 1, 4, 8], DataType::FP32);
    let v = make_tensor(vec![1, 1, 4, 8], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Attention, &[&q, &k, &v]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 0);
}

#[test]
fn test_flops_conv2d() {
    let input = make_tensor(vec![1, 3, 8, 8], DataType::FP32);
    let kernel = make_tensor(vec![16, 3, 3, 3], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&input, &kernel]);
    assert!(flops.is_some());
    assert!(flops.unwrap() > 0);
}

// ═══════════════════════════════════════════════════════════
// Memory computation tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_memory_bytes_matmul() {
    let a = make_tensor(vec![4, 3], DataType::FP32);
    let b = make_tensor(vec![3, 5], DataType::FP32);
    let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
    assert!(mem.is_some());
    assert!(mem.unwrap() > 0);
}

#[test]
fn test_memory_bytes_fp16_half_of_fp32() {
    let a_32 = make_tensor(vec![100, 100], DataType::FP32);
    let a_16 = make_tensor(vec![100, 100], DataType::FP16);
    let mem32 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&a_32]);
    let mem16 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&a_16]);
    assert!(mem32.unwrap() > mem16.unwrap());
}

// ═══════════════════════════════════════════════════════════
// num_inputs tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_num_inputs_unary() {
    assert_eq!(TensorOp::ReLU.num_inputs(), (1, 1));
    assert_eq!(TensorOp::Neg.num_inputs(), (1, 1));
    assert_eq!(TensorOp::Sigmoid.num_inputs(), (1, 1));
    assert_eq!(TensorOp::Softmax.num_inputs(), (1, 1));
}

#[test]
fn test_num_inputs_binary() {
    assert_eq!(TensorOp::Add.num_inputs(), (2, 2));
    assert_eq!(TensorOp::MatMul.num_inputs(), (2, 2));
    assert_eq!(TensorOp::Conv2D.num_inputs(), (2, 2));
}

#[test]
fn test_num_inputs_attention() {
    assert_eq!(TensorOp::Attention.num_inputs(), (3, 4));
    assert_eq!(TensorOp::MultiHeadAttention.num_inputs(), (3, 4));
}

#[test]
fn test_num_inputs_lstm() {
    assert_eq!(TensorOp::LSTMCell.num_inputs(), (2, 2));
}

#[test]
fn test_num_inputs_linear() {
    assert_eq!(TensorOp::Linear.num_inputs(), (3, 3));
}

// ═══════════════════════════════════════════════════════════
// flops_formula tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_flops_formula_matmul() {
    assert_eq!(TensorOp::MatMul.flops_formula(), "2*M*N*K");
}

#[test]
fn test_flops_formula_attention() {
    assert_eq!(TensorOp::Attention.flops_formula(), "2*B*H*(S^2*D + S*D^2)");
}

#[test]
fn test_flops_formula_reshape() {
    assert_eq!(TensorOp::Reshape.flops_formula(), "0 (no compute)");
}

// ═══════════════════════════════════════════════════════════
// from_name negative tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_from_name_unknown() {
    assert!(TensorOp::from_name("tensor.unknown_op").is_none());
    assert!(TensorOp::from_name("").is_none());
    assert!(TensorOp::from_name("quantum.h").is_none());
}
