use lift_core::context::Context;
use lift_core::pass::{Pass, PassResult, AnalysisCache};

/// FlashAttention pass: replaces standard attention with FlashAttention
/// when sequence length > threshold (default 512).
/// Same FLOPs, O(n) memory instead of O(n²).
#[derive(Debug)]
pub struct FlashAttentionPass {
    pub seq_len_threshold: u64,
}

impl Default for FlashAttentionPass {
    fn default() -> Self {
        Self { seq_len_threshold: 512 }
    }
}

impl Pass for FlashAttentionPass {
    fn name(&self) -> &str { "flash-attention" }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let mut replaced = 0usize;
        let flash_name = ctx.strings.intern("tensor.flash_attention");

        let op_keys: Vec<_> = ctx.ops.keys().collect();
        let attention_ops: Vec<_> = op_keys.into_iter().filter(|&ok| {
            if let Some(op) = ctx.ops.get(ok) {
                let name = ctx.strings.resolve(op.name);
                name == "tensor.attention" || name == "tensor.multi_head_attention"
            } else {
                false
            }
        }).collect();

        for op_key in attention_ops {
            // Check if seq_len attribute exceeds threshold
            let should_replace = if let Some(op) = ctx.ops.get(op_key) {
                let seq_len = op.attrs.get_integer("seq_len")
                    .map(|v| v as u64)
                    .unwrap_or(0);

                seq_len > self.seq_len_threshold
            } else {
                false
            };

            if should_replace {
                if let Some(op) = ctx.ops.get_mut(op_key) {
                    op.name = flash_name;
                    let causal = op.attrs.get_bool("causal").unwrap_or(false);
                    op.attrs.set("causal", lift_core::attributes::Attribute::Bool(causal));
                    replaced += 1;
                }
            }
        }

        if replaced > 0 {
            tracing::info!(
                pass = "flash-attention",
                replaced = replaced,
                threshold = self.seq_len_threshold,
                "FlashAttention replacement applied"
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
