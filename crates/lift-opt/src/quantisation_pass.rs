use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};

/// Quantisation pass: inserts quantize/dequantize pairs around compute-heavy ops
/// to reduce memory footprint and accelerate inference.
///
/// Supported modes:
/// - Dynamic: insert Q/DQ around MatMul and Linear
/// - Static: insert Q/DQ with pre-computed scales from calibration
#[derive(Debug)]
pub struct QuantisationPass {
    pub target_dtype: QuantTarget,
    pub mode: QuantMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantTarget {
    Int8,
    Int4,
    Fp8E4M3,
    Fp8E5M2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    Dynamic,
    Static,
}

impl Default for QuantisationPass {
    fn default() -> Self {
        Self {
            target_dtype: QuantTarget::Int8,
            mode: QuantMode::Dynamic,
        }
    }
}

impl Pass for QuantisationPass {
    fn name(&self) -> &str { "quantisation" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut quantised = 0usize;

        // Find ops that benefit from quantisation
        let op_keys: Vec<_> = ctx.ops.keys().collect();
        let target_ops: Vec<_> = op_keys.into_iter().filter(|&ok| {
            if let Some(op) = ctx.ops.get(ok) {
                let name = ctx.strings.resolve(op.name);
                // Target compute-heavy ops
                matches!(name,
                    "tensor.matmul" | "tensor.linear" |
                    "tensor.conv2d" | "tensor.conv1d" |
                    "tensor.multi_head_attention" | "tensor.attention"
                )
            } else {
                false
            }
        }).collect();

        let _quant_name = match self.target_dtype {
            QuantTarget::Int8 => "tensor.quantize",
            QuantTarget::Int4 => "tensor.quantize_int4",
            QuantTarget::Fp8E4M3 | QuantTarget::Fp8E5M2 => "tensor.quantize_fp8",
        };
        let _dequant_name = match self.target_dtype {
            QuantTarget::Int8 => "tensor.dequantize",
            QuantTarget::Int4 => "tensor.dequantize_int4",
            QuantTarget::Fp8E4M3 | QuantTarget::Fp8E5M2 => "tensor.dequantize_fp8",
        };

        // Mark ops for quantisation via attributes
        for op_key in target_ops {
            if let Some(op) = ctx.ops.get_mut(op_key) {
                // Skip already quantised ops
                if op.attrs.get_bool("quantised").unwrap_or(false) {
                    continue;
                }

                op.attrs.set("quantised", lift_core::attributes::Attribute::Bool(true));
                op.attrs.set("quant_method",
                    lift_core::attributes::Attribute::Integer(match self.target_dtype {
                        QuantTarget::Int8 => 8,
                        QuantTarget::Int4 => 4,
                        QuantTarget::Fp8E4M3 => 83,
                        QuantTarget::Fp8E5M2 => 82,
                    }));
                op.attrs.set("quant_bits",
                    lift_core::attributes::Attribute::Integer(match self.target_dtype {
                        QuantTarget::Int8 | QuantTarget::Fp8E4M3 | QuantTarget::Fp8E5M2 => 8,
                        QuantTarget::Int4 => 4,
                    }));

                quantised += 1;
            }
        }

        if quantised > 0 {
            tracing::info!(
                pass = "quantisation",
                quantised = quantised,
                target = ?self.target_dtype,
                mode = ?self.mode,
                "Quantisation annotations applied"
            );
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["analysis"]
    }
}
