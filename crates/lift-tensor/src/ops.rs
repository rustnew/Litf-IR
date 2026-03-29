use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    MatMul,
    Linear,
    Conv2D,
    Embedding,

    // Activations
    ReLU,
    GeLU,
    SiLU,
    Sigmoid,
    Softmax,
    Tanh,

    // Normalisation
    LayerNorm,
    RMSNorm,
    BatchNorm,

    // Shape operations
    Reshape,
    Transpose,
    Concat,
    Split,
    Gather,
    Scatter,

    // Constants
    Constant,
    Zeros,
    Ones,

    // Attention & LLM
    Attention,
    PagedAttention,
    MoEDispatch,
    MoECombine,

    // Quantisation
    Quantize,
    Dequantize,

    // Memory management
    Checkpoint,
    Offload,
    GradAccumulate,

    // Gradient operations
    GradMatMul,
    GradReLU,
    GradSoftmax,
    GradLayerNorm,
    GradAttention,

    // Parallelism
    ParallelSplit,
    ParallelAllReduce,
    PipelineSend,
    PipelineReceive,

    // Fused operations
    FusedMatMulBiasReLU,
    FusedMatMulBias,
    FusedLinearGeLU,
}

impl TensorOp {
    pub fn name(&self) -> &'static str {
        match self {
            TensorOp::Add => "tensor.add",
            TensorOp::Sub => "tensor.sub",
            TensorOp::Mul => "tensor.mul",
            TensorOp::Div => "tensor.div",
            TensorOp::Neg => "tensor.neg",
            TensorOp::MatMul => "tensor.matmul",
            TensorOp::Linear => "tensor.linear",
            TensorOp::Conv2D => "tensor.conv2d",
            TensorOp::Embedding => "tensor.embedding",
            TensorOp::ReLU => "tensor.relu",
            TensorOp::GeLU => "tensor.gelu",
            TensorOp::SiLU => "tensor.silu",
            TensorOp::Sigmoid => "tensor.sigmoid",
            TensorOp::Softmax => "tensor.softmax",
            TensorOp::Tanh => "tensor.tanh",
            TensorOp::LayerNorm => "tensor.layernorm",
            TensorOp::RMSNorm => "tensor.rmsnorm",
            TensorOp::BatchNorm => "tensor.batchnorm",
            TensorOp::Reshape => "tensor.reshape",
            TensorOp::Transpose => "tensor.transpose",
            TensorOp::Concat => "tensor.concat",
            TensorOp::Split => "tensor.split",
            TensorOp::Gather => "tensor.gather",
            TensorOp::Scatter => "tensor.scatter",
            TensorOp::Constant => "tensor.constant",
            TensorOp::Zeros => "tensor.zeros",
            TensorOp::Ones => "tensor.ones",
            TensorOp::Attention => "tensor.attention",
            TensorOp::PagedAttention => "tensor.paged_attention",
            TensorOp::MoEDispatch => "tensor.moe_dispatch",
            TensorOp::MoECombine => "tensor.moe_combine",
            TensorOp::Quantize => "tensor.quantize",
            TensorOp::Dequantize => "tensor.dequantize",
            TensorOp::Checkpoint => "tensor.checkpoint",
            TensorOp::Offload => "tensor.offload",
            TensorOp::GradAccumulate => "tensor.grad_accumulate",
            TensorOp::GradMatMul => "tensor.grad_matmul",
            TensorOp::GradReLU => "tensor.grad_relu",
            TensorOp::GradSoftmax => "tensor.grad_softmax",
            TensorOp::GradLayerNorm => "tensor.grad_layernorm",
            TensorOp::GradAttention => "tensor.grad_attention",
            TensorOp::ParallelSplit => "tensor.parallel_split",
            TensorOp::ParallelAllReduce => "tensor.parallel_allreduce",
            TensorOp::PipelineSend => "tensor.pipeline_send",
            TensorOp::PipelineReceive => "tensor.pipeline_receive",
            TensorOp::FusedMatMulBiasReLU => "tensor.fused_matmul_bias_relu",
            TensorOp::FusedMatMulBias => "tensor.fused_matmul_bias",
            TensorOp::FusedLinearGeLU => "tensor.fused_linear_gelu",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "tensor.add" => Some(TensorOp::Add),
            "tensor.sub" => Some(TensorOp::Sub),
            "tensor.mul" => Some(TensorOp::Mul),
            "tensor.div" => Some(TensorOp::Div),
            "tensor.neg" => Some(TensorOp::Neg),
            "tensor.matmul" => Some(TensorOp::MatMul),
            "tensor.linear" => Some(TensorOp::Linear),
            "tensor.conv2d" => Some(TensorOp::Conv2D),
            "tensor.embedding" => Some(TensorOp::Embedding),
            "tensor.relu" => Some(TensorOp::ReLU),
            "tensor.gelu" => Some(TensorOp::GeLU),
            "tensor.silu" => Some(TensorOp::SiLU),
            "tensor.sigmoid" => Some(TensorOp::Sigmoid),
            "tensor.softmax" => Some(TensorOp::Softmax),
            "tensor.tanh" => Some(TensorOp::Tanh),
            "tensor.layernorm" => Some(TensorOp::LayerNorm),
            "tensor.rmsnorm" => Some(TensorOp::RMSNorm),
            "tensor.batchnorm" => Some(TensorOp::BatchNorm),
            "tensor.reshape" => Some(TensorOp::Reshape),
            "tensor.transpose" => Some(TensorOp::Transpose),
            "tensor.concat" => Some(TensorOp::Concat),
            "tensor.split" => Some(TensorOp::Split),
            "tensor.gather" => Some(TensorOp::Gather),
            "tensor.scatter" => Some(TensorOp::Scatter),
            "tensor.constant" => Some(TensorOp::Constant),
            "tensor.zeros" => Some(TensorOp::Zeros),
            "tensor.ones" => Some(TensorOp::Ones),
            "tensor.attention" => Some(TensorOp::Attention),
            "tensor.paged_attention" => Some(TensorOp::PagedAttention),
            "tensor.moe_dispatch" => Some(TensorOp::MoEDispatch),
            "tensor.moe_combine" => Some(TensorOp::MoECombine),
            "tensor.quantize" => Some(TensorOp::Quantize),
            "tensor.dequantize" => Some(TensorOp::Dequantize),
            "tensor.checkpoint" => Some(TensorOp::Checkpoint),
            "tensor.offload" => Some(TensorOp::Offload),
            "tensor.grad_accumulate" => Some(TensorOp::GradAccumulate),
            "tensor.grad_matmul" => Some(TensorOp::GradMatMul),
            "tensor.grad_relu" => Some(TensorOp::GradReLU),
            "tensor.grad_softmax" => Some(TensorOp::GradSoftmax),
            "tensor.grad_layernorm" => Some(TensorOp::GradLayerNorm),
            "tensor.grad_attention" => Some(TensorOp::GradAttention),
            "tensor.parallel_split" => Some(TensorOp::ParallelSplit),
            "tensor.parallel_allreduce" => Some(TensorOp::ParallelAllReduce),
            "tensor.pipeline_send" => Some(TensorOp::PipelineSend),
            "tensor.pipeline_receive" => Some(TensorOp::PipelineReceive),
            "tensor.fused_matmul_bias_relu" => Some(TensorOp::FusedMatMulBiasReLU),
            "tensor.fused_matmul_bias" => Some(TensorOp::FusedMatMulBias),
            "tensor.fused_linear_gelu" => Some(TensorOp::FusedLinearGeLU),
            _ => None,
        }
    }

    pub fn num_inputs(&self) -> (usize, usize) {
        match self {
            TensorOp::Neg | TensorOp::ReLU | TensorOp::GeLU | TensorOp::SiLU |
            TensorOp::Sigmoid | TensorOp::Tanh | TensorOp::Reshape |
            TensorOp::Transpose | TensorOp::Quantize | TensorOp::Dequantize |
            TensorOp::Offload | TensorOp::GradReLU => (1, 1),

            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div |
            TensorOp::MatMul | TensorOp::GradMatMul => (2, 2),

            TensorOp::Linear | TensorOp::FusedMatMulBias |
            TensorOp::FusedLinearGeLU => (3, 3),

            TensorOp::FusedMatMulBiasReLU => (3, 3),

            TensorOp::LayerNorm | TensorOp::RMSNorm => (2, 3),
            TensorOp::BatchNorm => (3, 5),

            TensorOp::Softmax => (1, 1),

            TensorOp::Attention => (3, 4), // Q, K, V, optional mask
            TensorOp::PagedAttention => (3, 5),

            TensorOp::Conv2D => (2, 3),
            TensorOp::Embedding => (2, 2),

            TensorOp::Constant | TensorOp::Zeros | TensorOp::Ones => (0, 0),

            _ => (0, usize::MAX),
        }
    }

    pub fn flops_formula(&self) -> &'static str {
        match self {
            TensorOp::MatMul => "2*M*N*K",
            TensorOp::Linear => "2*M*N*K + N (bias)",
            TensorOp::Add | TensorOp::Sub | TensorOp::Mul | TensorOp::Div => "N (element count)",
            TensorOp::ReLU | TensorOp::Sigmoid | TensorOp::Tanh => "N (element count)",
            TensorOp::GeLU | TensorOp::SiLU => "~5*N",
            TensorOp::Softmax => "5*N (exp + sum + div)",
            TensorOp::LayerNorm | TensorOp::RMSNorm => "5*N",
            TensorOp::Conv2D => "2*Cout*Cin*Kh*Kw*Oh*Ow",
            TensorOp::Attention => "4*B*H*S*S*D (standard) or 4*B*H*S*D (flash)",
            TensorOp::Reshape | TensorOp::Transpose => "0 (no compute)",
            _ => "varies",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_name_roundtrip() {
        for op in &[TensorOp::MatMul, TensorOp::ReLU, TensorOp::Attention, TensorOp::Softmax] {
            let name = op.name();
            let recovered = TensorOp::from_name(name).unwrap();
            assert_eq!(op, &recovered);
        }
    }

    #[test]
    fn test_all_ops_have_names() {
        let ops = vec![
            TensorOp::Add, TensorOp::Sub, TensorOp::Mul, TensorOp::Div,
            TensorOp::MatMul, TensorOp::Linear, TensorOp::ReLU, TensorOp::GeLU,
            TensorOp::Softmax, TensorOp::LayerNorm, TensorOp::Attention,
        ];
        for op in ops {
            assert!(!op.name().is_empty());
            assert!(TensorOp::from_name(op.name()).is_some());
        }
    }
}
