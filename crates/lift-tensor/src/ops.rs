use serde::{Serialize, Deserialize};

/// FP8 quantisation format variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

/// Aggregation type for GNN message passing and pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregationType {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorOp {
    // ── Arithmetic ──
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    MatMul,
    Linear,
    Conv2D,
    Embedding,

    // ── Activations ──
    ReLU,
    GeLU,
    SiLU,
    Sigmoid,
    Softmax,
    Tanh,
    LeakyReLU,
    ELU,
    Mish,
    HardSwish,
    HardSigmoid,

    // ── Normalisation ──
    LayerNorm,
    RMSNorm,
    BatchNorm,
    GroupNorm,
    InstanceNorm,

    // ── Shape operations ──
    Reshape,
    Transpose,
    Concat,
    Split,
    Gather,
    Scatter,
    Squeeze,
    Unsqueeze,
    Permute,
    Expand,
    Slice,
    Pad,
    Tile,

    // ── Constants ──
    Constant,
    Zeros,
    Ones,
    Arange,
    Full,

    // ── Attention variants ──
    Attention,
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    FlashAttention,
    SlidingWindowAttention,
    CrossAttention,
    PagedAttention,

    // ── MoE (Mixture of Experts) ──
    MoEDispatch,
    MoECombine,

    // ── Convolution variants ──
    Conv1D,
    Conv3D,
    ConvTranspose2D,
    DepthwiseConv2D,
    DilatedConv2D,

    // ── Pooling ──
    MaxPool2D,
    AvgPool2D,
    AdaptiveAvgPool2D,
    GlobalAvgPool,

    // ── Recurrent ──
    LSTMCell,
    GRUCell,
    RNNCell,

    // ── Advanced math ──
    Einsum,
    FFT,
    IFFT,
    SVD,
    Eig,
    Solve,
    TopK,
    Sort,
    Cumsum,
    Where,
    Clamp,

    // ── Sparse ──
    SparseMatMul,
    SparseEmbedding,

    // ── Quantisation ──
    Quantize,
    Dequantize,
    QuantizeInt4,
    DequantizeInt4,
    QuantizeFp8,
    DequantizeFp8,

    // ── Diffusion / Generative ──
    UNetDownBlock,
    UNetUpBlock,
    TimestepEmbedding,

    // ── GNN (Graph Neural Networks) ──
    GNNMessagePassing,
    GNNGlobalPooling,

    // ── Memory management ──
    Checkpoint,
    Offload,
    GradAccumulate,

    // ── Gradient operations ──
    GradMatMul,
    GradReLU,
    GradSoftmax,
    GradLayerNorm,
    GradAttention,
    GradConv2D,
    GradLinear,
    GradGeLU,

    // ── Parallelism ──
    ParallelSplit,
    ParallelAllReduce,
    PipelineSend,
    PipelineReceive,

    // ── Fused operations ──
    FusedMatMulBiasReLU,
    FusedMatMulBias,
    FusedLinearGeLU,
    FusedAttentionLayerNorm,
    FusedLinearSiLU,
    FusedConvBatchNormReLU,
}

impl TensorOp {
    pub fn name(&self) -> &'static str {
        match self {
            // Arithmetic
            Self::Add => "tensor.add",
            Self::Sub => "tensor.sub",
            Self::Mul => "tensor.mul",
            Self::Div => "tensor.div",
            Self::Neg => "tensor.neg",
            Self::MatMul => "tensor.matmul",
            Self::Linear => "tensor.linear",
            Self::Conv2D => "tensor.conv2d",
            Self::Embedding => "tensor.embedding",
            // Activations
            Self::ReLU => "tensor.relu",
            Self::GeLU => "tensor.gelu",
            Self::SiLU => "tensor.silu",
            Self::Sigmoid => "tensor.sigmoid",
            Self::Softmax => "tensor.softmax",
            Self::Tanh => "tensor.tanh",
            Self::LeakyReLU => "tensor.leaky_relu",
            Self::ELU => "tensor.elu",
            Self::Mish => "tensor.mish",
            Self::HardSwish => "tensor.hard_swish",
            Self::HardSigmoid => "tensor.hard_sigmoid",
            // Normalisation
            Self::LayerNorm => "tensor.layernorm",
            Self::RMSNorm => "tensor.rmsnorm",
            Self::BatchNorm => "tensor.batchnorm",
            Self::GroupNorm => "tensor.groupnorm",
            Self::InstanceNorm => "tensor.instancenorm",
            // Shape
            Self::Reshape => "tensor.reshape",
            Self::Transpose => "tensor.transpose",
            Self::Concat => "tensor.concat",
            Self::Split => "tensor.split",
            Self::Gather => "tensor.gather",
            Self::Scatter => "tensor.scatter",
            Self::Squeeze => "tensor.squeeze",
            Self::Unsqueeze => "tensor.unsqueeze",
            Self::Permute => "tensor.permute",
            Self::Expand => "tensor.expand",
            Self::Slice => "tensor.slice",
            Self::Pad => "tensor.pad",
            Self::Tile => "tensor.tile",
            // Constants
            Self::Constant => "tensor.constant",
            Self::Zeros => "tensor.zeros",
            Self::Ones => "tensor.ones",
            Self::Arange => "tensor.arange",
            Self::Full => "tensor.full",
            // Attention variants
            Self::Attention => "tensor.attention",
            Self::MultiHeadAttention => "tensor.multi_head_attention",
            Self::MultiQueryAttention => "tensor.multi_query_attention",
            Self::GroupedQueryAttention => "tensor.grouped_query_attention",
            Self::FlashAttention => "tensor.flash_attention",
            Self::SlidingWindowAttention => "tensor.sliding_window_attention",
            Self::CrossAttention => "tensor.cross_attention",
            Self::PagedAttention => "tensor.paged_attention",
            // MoE
            Self::MoEDispatch => "tensor.moe_dispatch",
            Self::MoECombine => "tensor.moe_combine",
            // Conv variants
            Self::Conv1D => "tensor.conv1d",
            Self::Conv3D => "tensor.conv3d",
            Self::ConvTranspose2D => "tensor.conv_transpose2d",
            Self::DepthwiseConv2D => "tensor.depthwise_conv2d",
            Self::DilatedConv2D => "tensor.dilated_conv2d",
            // Pooling
            Self::MaxPool2D => "tensor.maxpool2d",
            Self::AvgPool2D => "tensor.avgpool2d",
            Self::AdaptiveAvgPool2D => "tensor.adaptive_avgpool2d",
            Self::GlobalAvgPool => "tensor.global_avgpool",
            // Recurrent
            Self::LSTMCell => "tensor.lstm_cell",
            Self::GRUCell => "tensor.gru_cell",
            Self::RNNCell => "tensor.rnn_cell",
            // Advanced math
            Self::Einsum => "tensor.einsum",
            Self::FFT => "tensor.fft",
            Self::IFFT => "tensor.ifft",
            Self::SVD => "tensor.svd",
            Self::Eig => "tensor.eig",
            Self::Solve => "tensor.solve",
            Self::TopK => "tensor.topk",
            Self::Sort => "tensor.sort",
            Self::Cumsum => "tensor.cumsum",
            Self::Where => "tensor.where",
            Self::Clamp => "tensor.clamp",
            // Sparse
            Self::SparseMatMul => "tensor.sparse_matmul",
            Self::SparseEmbedding => "tensor.sparse_embedding",
            // Quantisation
            Self::Quantize => "tensor.quantize",
            Self::Dequantize => "tensor.dequantize",
            Self::QuantizeInt4 => "tensor.quantize_int4",
            Self::DequantizeInt4 => "tensor.dequantize_int4",
            Self::QuantizeFp8 => "tensor.quantize_fp8",
            Self::DequantizeFp8 => "tensor.dequantize_fp8",
            // Diffusion / Generative
            Self::UNetDownBlock => "tensor.unet_down_block",
            Self::UNetUpBlock => "tensor.unet_up_block",
            Self::TimestepEmbedding => "tensor.timestep_embedding",
            // GNN
            Self::GNNMessagePassing => "tensor.gnn_message_passing",
            Self::GNNGlobalPooling => "tensor.gnn_global_pooling",
            // Memory management
            Self::Checkpoint => "tensor.checkpoint",
            Self::Offload => "tensor.offload",
            Self::GradAccumulate => "tensor.grad_accumulate",
            // Gradient operations
            Self::GradMatMul => "tensor.grad_matmul",
            Self::GradReLU => "tensor.grad_relu",
            Self::GradSoftmax => "tensor.grad_softmax",
            Self::GradLayerNorm => "tensor.grad_layernorm",
            Self::GradAttention => "tensor.grad_attention",
            Self::GradConv2D => "tensor.grad_conv2d",
            Self::GradLinear => "tensor.grad_linear",
            Self::GradGeLU => "tensor.grad_gelu",
            // Parallelism
            Self::ParallelSplit => "tensor.parallel_split",
            Self::ParallelAllReduce => "tensor.parallel_allreduce",
            Self::PipelineSend => "tensor.pipeline_send",
            Self::PipelineReceive => "tensor.pipeline_receive",
            // Fused operations
            Self::FusedMatMulBiasReLU => "tensor.fused_matmul_bias_relu",
            Self::FusedMatMulBias => "tensor.fused_matmul_bias",
            Self::FusedLinearGeLU => "tensor.fused_linear_gelu",
            Self::FusedAttentionLayerNorm => "tensor.fused_attention_layernorm",
            Self::FusedLinearSiLU => "tensor.fused_linear_silu",
            Self::FusedConvBatchNormReLU => "tensor.fused_conv_batchnorm_relu",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "tensor.add" => Some(Self::Add),
            "tensor.sub" => Some(Self::Sub),
            "tensor.mul" => Some(Self::Mul),
            "tensor.div" => Some(Self::Div),
            "tensor.neg" => Some(Self::Neg),
            "tensor.matmul" => Some(Self::MatMul),
            "tensor.linear" => Some(Self::Linear),
            "tensor.conv2d" => Some(Self::Conv2D),
            "tensor.embedding" => Some(Self::Embedding),
            "tensor.relu" => Some(Self::ReLU),
            "tensor.gelu" => Some(Self::GeLU),
            "tensor.silu" => Some(Self::SiLU),
            "tensor.sigmoid" => Some(Self::Sigmoid),
            "tensor.softmax" => Some(Self::Softmax),
            "tensor.tanh" => Some(Self::Tanh),
            "tensor.leaky_relu" => Some(Self::LeakyReLU),
            "tensor.elu" => Some(Self::ELU),
            "tensor.mish" => Some(Self::Mish),
            "tensor.hard_swish" => Some(Self::HardSwish),
            "tensor.hard_sigmoid" => Some(Self::HardSigmoid),
            "tensor.layernorm" => Some(Self::LayerNorm),
            "tensor.rmsnorm" => Some(Self::RMSNorm),
            "tensor.batchnorm" => Some(Self::BatchNorm),
            "tensor.groupnorm" => Some(Self::GroupNorm),
            "tensor.instancenorm" => Some(Self::InstanceNorm),
            "tensor.reshape" => Some(Self::Reshape),
            "tensor.transpose" => Some(Self::Transpose),
            "tensor.concat" => Some(Self::Concat),
            "tensor.split" => Some(Self::Split),
            "tensor.gather" => Some(Self::Gather),
            "tensor.scatter" => Some(Self::Scatter),
            "tensor.squeeze" => Some(Self::Squeeze),
            "tensor.unsqueeze" => Some(Self::Unsqueeze),
            "tensor.permute" => Some(Self::Permute),
            "tensor.expand" => Some(Self::Expand),
            "tensor.slice" => Some(Self::Slice),
            "tensor.pad" => Some(Self::Pad),
            "tensor.tile" => Some(Self::Tile),
            "tensor.constant" => Some(Self::Constant),
            "tensor.zeros" => Some(Self::Zeros),
            "tensor.ones" => Some(Self::Ones),
            "tensor.arange" => Some(Self::Arange),
            "tensor.full" => Some(Self::Full),
            "tensor.attention" => Some(Self::Attention),
            "tensor.multi_head_attention" => Some(Self::MultiHeadAttention),
            "tensor.multi_query_attention" => Some(Self::MultiQueryAttention),
            "tensor.grouped_query_attention" => Some(Self::GroupedQueryAttention),
            "tensor.flash_attention" => Some(Self::FlashAttention),
            "tensor.sliding_window_attention" => Some(Self::SlidingWindowAttention),
            "tensor.cross_attention" => Some(Self::CrossAttention),
            "tensor.paged_attention" => Some(Self::PagedAttention),
            "tensor.moe_dispatch" => Some(Self::MoEDispatch),
            "tensor.moe_combine" => Some(Self::MoECombine),
            "tensor.conv1d" => Some(Self::Conv1D),
            "tensor.conv3d" => Some(Self::Conv3D),
            "tensor.conv_transpose2d" => Some(Self::ConvTranspose2D),
            "tensor.depthwise_conv2d" => Some(Self::DepthwiseConv2D),
            "tensor.dilated_conv2d" => Some(Self::DilatedConv2D),
            "tensor.maxpool2d" => Some(Self::MaxPool2D),
            "tensor.avgpool2d" => Some(Self::AvgPool2D),
            "tensor.adaptive_avgpool2d" => Some(Self::AdaptiveAvgPool2D),
            "tensor.global_avgpool" => Some(Self::GlobalAvgPool),
            "tensor.lstm_cell" => Some(Self::LSTMCell),
            "tensor.gru_cell" => Some(Self::GRUCell),
            "tensor.rnn_cell" => Some(Self::RNNCell),
            "tensor.einsum" => Some(Self::Einsum),
            "tensor.fft" => Some(Self::FFT),
            "tensor.ifft" => Some(Self::IFFT),
            "tensor.svd" => Some(Self::SVD),
            "tensor.eig" => Some(Self::Eig),
            "tensor.solve" => Some(Self::Solve),
            "tensor.topk" => Some(Self::TopK),
            "tensor.sort" => Some(Self::Sort),
            "tensor.cumsum" => Some(Self::Cumsum),
            "tensor.where" => Some(Self::Where),
            "tensor.clamp" => Some(Self::Clamp),
            "tensor.sparse_matmul" => Some(Self::SparseMatMul),
            "tensor.sparse_embedding" => Some(Self::SparseEmbedding),
            "tensor.quantize" => Some(Self::Quantize),
            "tensor.dequantize" => Some(Self::Dequantize),
            "tensor.quantize_int4" => Some(Self::QuantizeInt4),
            "tensor.dequantize_int4" => Some(Self::DequantizeInt4),
            "tensor.quantize_fp8" => Some(Self::QuantizeFp8),
            "tensor.dequantize_fp8" => Some(Self::DequantizeFp8),
            "tensor.unet_down_block" => Some(Self::UNetDownBlock),
            "tensor.unet_up_block" => Some(Self::UNetUpBlock),
            "tensor.timestep_embedding" => Some(Self::TimestepEmbedding),
            "tensor.gnn_message_passing" => Some(Self::GNNMessagePassing),
            "tensor.gnn_global_pooling" => Some(Self::GNNGlobalPooling),
            "tensor.checkpoint" => Some(Self::Checkpoint),
            "tensor.offload" => Some(Self::Offload),
            "tensor.grad_accumulate" => Some(Self::GradAccumulate),
            "tensor.grad_matmul" => Some(Self::GradMatMul),
            "tensor.grad_relu" => Some(Self::GradReLU),
            "tensor.grad_softmax" => Some(Self::GradSoftmax),
            "tensor.grad_layernorm" => Some(Self::GradLayerNorm),
            "tensor.grad_attention" => Some(Self::GradAttention),
            "tensor.grad_conv2d" => Some(Self::GradConv2D),
            "tensor.grad_linear" => Some(Self::GradLinear),
            "tensor.grad_gelu" => Some(Self::GradGeLU),
            "tensor.parallel_split" => Some(Self::ParallelSplit),
            "tensor.parallel_allreduce" => Some(Self::ParallelAllReduce),
            "tensor.pipeline_send" => Some(Self::PipelineSend),
            "tensor.pipeline_receive" => Some(Self::PipelineReceive),
            "tensor.fused_matmul_bias_relu" => Some(Self::FusedMatMulBiasReLU),
            "tensor.fused_matmul_bias" => Some(Self::FusedMatMulBias),
            "tensor.fused_linear_gelu" => Some(Self::FusedLinearGeLU),
            "tensor.fused_attention_layernorm" => Some(Self::FusedAttentionLayerNorm),
            "tensor.fused_linear_silu" => Some(Self::FusedLinearSiLU),
            "tensor.fused_conv_batchnorm_relu" => Some(Self::FusedConvBatchNormReLU),
            _ => None,
        }
    }

    pub fn num_inputs(&self) -> (usize, usize) {
        match self {
            // Unary (1 input)
            Self::Neg | Self::ReLU | Self::GeLU | Self::SiLU |
            Self::Sigmoid | Self::Tanh | Self::LeakyReLU | Self::ELU |
            Self::Mish | Self::HardSwish | Self::HardSigmoid |
            Self::Reshape | Self::Transpose | Self::Squeeze | Self::Unsqueeze |
            Self::Permute | Self::Expand | Self::Slice | Self::Pad | Self::Tile |
            Self::Quantize | Self::Dequantize |
            Self::QuantizeInt4 | Self::DequantizeInt4 |
            Self::QuantizeFp8 | Self::DequantizeFp8 |
            Self::Offload | Self::Checkpoint |
            Self::GradReLU | Self::GradGeLU |
            Self::Softmax | Self::Cumsum | Self::Sort | Self::TopK |
            Self::FFT | Self::IFFT | Self::SVD | Self::Eig |
            Self::GlobalAvgPool | Self::AdaptiveAvgPool2D |
            Self::GNNGlobalPooling => (1, 1),

            // Binary (2 inputs)
            Self::Add | Self::Sub | Self::Mul | Self::Div |
            Self::MatMul | Self::SparseMatMul |
            Self::GradMatMul | Self::Embedding | Self::SparseEmbedding |
            Self::Conv2D | Self::Conv1D | Self::Conv3D |
            Self::ConvTranspose2D | Self::DepthwiseConv2D | Self::DilatedConv2D |
            Self::MaxPool2D | Self::AvgPool2D |
            Self::Solve | Self::GradConv2D | Self::Concat => (2, 2),

            // Ternary (3 inputs)
            Self::Linear | Self::FusedMatMulBias | Self::FusedLinearGeLU |
            Self::FusedMatMulBiasReLU | Self::FusedLinearSiLU |
            Self::Where | Self::Clamp |
            Self::GradLinear => (3, 3),

            // Attention (3-4 inputs: Q, K, V, optional mask)
            Self::Attention | Self::MultiHeadAttention |
            Self::MultiQueryAttention | Self::GroupedQueryAttention |
            Self::FlashAttention | Self::SlidingWindowAttention |
            Self::CrossAttention | Self::GradAttention => (3, 4),
            Self::PagedAttention => (3, 5),
            Self::FusedAttentionLayerNorm => (3, 5),

            // Normalisation (variable: input + scale + bias)
            Self::LayerNorm | Self::RMSNorm | Self::GroupNorm |
            Self::InstanceNorm | Self::GradLayerNorm => (2, 3),
            Self::BatchNorm | Self::FusedConvBatchNormReLU => (3, 5),

            // Recurrent (2 inputs: input, hidden state)
            Self::LSTMCell | Self::GRUCell | Self::RNNCell => (2, 2),

            // GNN (2 inputs: node features, edge index)
            Self::GNNMessagePassing => (2, 3),

            // Diffusion blocks (2-3 inputs)
            Self::UNetDownBlock | Self::UNetUpBlock => (2, 3),
            Self::TimestepEmbedding => (1, 1),

            // MoE
            Self::MoEDispatch => (2, 3),
            Self::MoECombine => (2, 3),

            // Constants (0 inputs)
            Self::Constant | Self::Zeros | Self::Ones |
            Self::Arange | Self::Full => (0, 0),

            // Einsum (variable)
            Self::Einsum => (1, usize::MAX),

            // Parallelism / memory
            Self::GradAccumulate | Self::GradSoftmax |
            Self::ParallelSplit | Self::ParallelAllReduce |
            Self::PipelineSend | Self::PipelineReceive |
            Self::Gather | Self::Scatter | Self::Split => (1, usize::MAX),
        }
    }

    /// Returns the asymptotic FLOPs formula as a human-readable string.
    pub fn flops_formula(&self) -> &'static str {
        match self {
            Self::MatMul | Self::SparseMatMul => "2*M*N*K",
            Self::Linear => "2*M*N*K + N (bias)",
            Self::Add | Self::Sub | Self::Mul | Self::Div => "N (element count)",
            Self::ReLU | Self::Sigmoid | Self::Tanh |
            Self::LeakyReLU | Self::ELU | Self::HardSigmoid => "N",
            Self::GeLU | Self::SiLU | Self::Mish | Self::HardSwish => "~8*N",
            Self::Softmax => "5*N (exp + sum + div)",
            Self::LayerNorm | Self::RMSNorm |
            Self::GroupNorm | Self::InstanceNorm => "7*N",
            Self::BatchNorm => "5*N",
            Self::Conv2D | Self::DepthwiseConv2D | Self::DilatedConv2D => "2*Cout*Cin*Kh*Kw*Oh*Ow",
            Self::Conv1D => "2*Cout*Cin*K*Oout",
            Self::Conv3D => "2*Cout*Cin*Kd*Kh*Kw*Od*Oh*Ow",
            Self::Attention | Self::MultiHeadAttention |
            Self::GroupedQueryAttention | Self::MultiQueryAttention |
            Self::FlashAttention | Self::SlidingWindowAttention |
            Self::CrossAttention => "2*B*H*(S^2*D + S*D^2)",
            Self::LSTMCell => "4*(input_size+hidden)*hidden*2",
            Self::GRUCell => "3*(input_size+hidden)*hidden*2",
            Self::RNNCell => "(input_size+hidden)*hidden*2",
            Self::FFT | Self::IFFT => "5*N*log2(N)",
            Self::Einsum => "depends on equation",
            Self::MaxPool2D | Self::AvgPool2D | Self::AdaptiveAvgPool2D |
            Self::GlobalAvgPool => "N (comparisons or additions)",
            Self::Reshape | Self::Transpose | Self::Squeeze | Self::Unsqueeze |
            Self::Permute | Self::Expand | Self::Slice | Self::Pad | Self::Tile |
            Self::Concat | Self::Split | Self::Gather | Self::Scatter => "0 (no compute)",
            _ => "varies",
        }
    }

    /// Returns `true` if this op performs no arithmetic (zero FLOPs).
    #[inline]
    pub fn is_zero_flop(&self) -> bool {
        matches!(self,
            Self::Reshape | Self::Transpose | Self::Squeeze | Self::Unsqueeze |
            Self::Permute | Self::Expand | Self::Slice | Self::Pad | Self::Tile |
            Self::Concat | Self::Split | Self::Gather | Self::Scatter |
            Self::Constant | Self::Zeros | Self::Ones | Self::Arange | Self::Full |
            Self::Checkpoint | Self::Offload |
            Self::PipelineSend | Self::PipelineReceive |
            Self::ParallelSplit | Self::ParallelAllReduce
        )
    }

    /// Returns `true` if this is an element-wise (unary or binary) activation.
    #[inline]
    pub fn is_activation(&self) -> bool {
        matches!(self,
            Self::ReLU | Self::GeLU | Self::SiLU | Self::Sigmoid | Self::Tanh |
            Self::LeakyReLU | Self::ELU | Self::Mish |
            Self::HardSwish | Self::HardSigmoid
        )
    }

    /// Returns `true` if this is an attention variant.
    #[inline]
    pub fn is_attention(&self) -> bool {
        matches!(self,
            Self::Attention | Self::MultiHeadAttention | Self::MultiQueryAttention |
            Self::GroupedQueryAttention | Self::FlashAttention |
            Self::SlidingWindowAttention | Self::CrossAttention | Self::PagedAttention
        )
    }

    /// Returns `true` if this is a convolution variant.
    #[inline]
    pub fn is_convolution(&self) -> bool {
        matches!(self,
            Self::Conv1D | Self::Conv2D | Self::Conv3D |
            Self::ConvTranspose2D | Self::DepthwiseConv2D | Self::DilatedConv2D
        )
    }

    /// Returns `true` if this is a normalisation op.
    #[inline]
    pub fn is_normalisation(&self) -> bool {
        matches!(self,
            Self::LayerNorm | Self::RMSNorm | Self::BatchNorm |
            Self::GroupNorm | Self::InstanceNorm
        )
    }

    /// Returns `true` if this is a fused operation.
    #[inline]
    pub fn is_fused(&self) -> bool {
        matches!(self,
            Self::FusedMatMulBiasReLU | Self::FusedMatMulBias |
            Self::FusedLinearGeLU | Self::FusedAttentionLayerNorm |
            Self::FusedLinearSiLU | Self::FusedConvBatchNormReLU
        )
    }

    /// Returns `true` if this is a gradient (backward) operation.
    #[inline]
    pub fn is_gradient(&self) -> bool {
        matches!(self,
            Self::GradMatMul | Self::GradReLU | Self::GradSoftmax |
            Self::GradLayerNorm | Self::GradAttention |
            Self::GradConv2D | Self::GradLinear | Self::GradGeLU
        )
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
