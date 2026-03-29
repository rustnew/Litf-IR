use lift_core::context::Context;
use lift_core::blocks::BlockKey;
use lift_core::types::{CoreType, TypeData, TensorTypeInfo};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub total_flops: u64,
    pub total_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub num_ops: usize,
    pub num_tensor_ops: usize,
    pub num_quantum_ops: usize,
    pub num_hybrid_ops: usize,
    pub op_breakdown: std::collections::HashMap<String, usize>,
    pub warnings: Vec<String>,
}

pub fn analyze_module(ctx: &Context) -> AnalysisReport {
    let mut report = AnalysisReport::default();

    for (_op_key, op) in &ctx.ops {
        report.num_ops += 1;
        let op_name = ctx.strings.resolve(op.name).to_string();
        let dialect = ctx.strings.resolve(op.dialect).to_string();

        *report.op_breakdown.entry(op_name.clone()).or_insert(0) += 1;

        match dialect.as_str() {
            "tensor" => {
                report.num_tensor_ops += 1;
                // Compute FLOPS for tensor ops
                let input_infos: Vec<&TensorTypeInfo> = op.inputs.iter()
                    .filter_map(|&v| ctx.get_value(v))
                    .filter_map(|v| ctx.get_tensor_info(v.ty))
                    .collect();

                if let Some(tensor_op) = lift_tensor::TensorOp::from_name(&op_name) {
                    let input_refs: Vec<&TensorTypeInfo> = input_infos.iter().copied().collect();
                    if let Some(flops) = lift_tensor::ShapeInference::compute_flops(&tensor_op, &input_refs) {
                        report.total_flops += flops;
                    }
                    if let Some(mem) = lift_tensor::ShapeInference::compute_memory_bytes(&tensor_op, &input_refs) {
                        report.total_memory_bytes += mem;
                    }
                }
            }
            "quantum" => {
                report.num_quantum_ops += 1;
            }
            "hybrid" => {
                report.num_hybrid_ops += 1;
            }
            _ => {}
        }
    }

    report.peak_memory_bytes = estimate_peak_memory(ctx);
    report
}

fn estimate_peak_memory(ctx: &Context) -> u64 {
    let mut total: u64 = 0;
    for (_val_key, val) in &ctx.values {
        if let CoreType::Opaque { data: TypeData::Tensor(info), .. } = ctx.resolve_type(val.ty) {
            if let Some(bytes) = tensor_size_bytes(info) {
                total += bytes as u64;
            }
        }
    }
    total
}

fn tensor_size_bytes(info: &TensorTypeInfo) -> Option<usize> {
    let mut count = 1usize;
    for dim in &info.shape {
        count = count.checked_mul(dim.static_value()?)?;
    }
    Some(count * info.dtype.byte_size())
}

pub fn analyze_block(ctx: &Context, block_key: BlockKey) -> AnalysisReport {
    let mut report = AnalysisReport::default();

    if let Some(block) = ctx.get_block(block_key) {
        for &op_key in &block.ops {
            report.num_ops += 1;
            if let Some(op) = ctx.get_op(op_key) {
                let op_name = ctx.strings.resolve(op.name).to_string();
                *report.op_breakdown.entry(op_name).or_insert(0) += 1;
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_analysis() {
        let ctx = Context::new();
        let report = analyze_module(&ctx);
        assert_eq!(report.num_ops, 0);
        assert_eq!(report.total_flops, 0);
    }
}
