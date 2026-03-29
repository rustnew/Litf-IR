use serde::{Serialize, Deserialize};
use lift_core::types::{DataType, Dimension, MemoryLayout};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorType {
    Tensor {
        shape: Vec<Dimension>,
        dtype: DataType,
        layout: MemoryLayout,
    },
    AttentionTensor {
        batch: Dimension,
        seq_len: Dimension,
        num_heads: usize,
        head_dim: usize,
        dtype: DataType,
    },
    KVCache {
        max_seq: Dimension,
        num_heads: usize,
        head_dim: usize,
        dtype: DataType,
        is_paged: bool,
    },
    SparseTensor {
        num_experts: usize,
        capacity: usize,
        dtype: DataType,
    },
}

impl TensorType {
    pub fn dtype(&self) -> DataType {
        match self {
            TensorType::Tensor { dtype, .. } => *dtype,
            TensorType::AttentionTensor { dtype, .. } => *dtype,
            TensorType::KVCache { dtype, .. } => *dtype,
            TensorType::SparseTensor { dtype, .. } => *dtype,
        }
    }

    pub fn element_count(&self) -> Option<usize> {
        match self {
            TensorType::Tensor { shape, .. } => {
                let mut count = 1usize;
                for dim in shape {
                    match dim.static_value() {
                        Some(v) => count = count.checked_mul(v)?,
                        None => return None,
                    }
                }
                Some(count)
            }
            TensorType::AttentionTensor { batch, seq_len, num_heads, head_dim, .. } => {
                let b = batch.static_value()?;
                let s = seq_len.static_value()?;
                Some(b * s * num_heads * head_dim)
            }
            _ => None,
        }
    }

    pub fn size_bytes(&self) -> Option<usize> {
        Some(self.element_count()? * self.dtype().byte_size())
    }

    pub fn rank(&self) -> Option<usize> {
        match self {
            TensorType::Tensor { shape, .. } => Some(shape.len()),
            TensorType::AttentionTensor { .. } => Some(4), // [batch, seq, heads, dim]
            TensorType::KVCache { .. } => Some(4),
            TensorType::SparseTensor { .. } => Some(3),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionImpl {
    Standard,
    FlashAttentionV2,
    FlashAttentionV3,
    PagedAttention,
    SDPA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    DynamicINT8,
    StaticINT8,
    FP8E4M3,
    FP8E5M2,
    INT4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismStrategy {
    DataParallel,
    TensorParallel,
    PipelineParallel,
    SequenceParallel,
}
