use lift_opt::{
    RotationMerge, FlashAttentionPass, CommonSubexprElimination,
    QuantisationPass, NoiseAwareSchedule, LayoutMapping,
};
use lift_opt::quantisation_pass::{QuantTarget, QuantMode};
use lift_core::pass::{Pass, PassResult, AnalysisCache};
use lift_core::context::Context;
use lift_core::attributes::Attribute;

// ═══════════════════════════════════════════════════════════
// RotationMerge pass tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_rotation_merge_name() {
    let pass = RotationMerge;
    assert_eq!(pass.name(), "rotation-merge");
}

#[test]
fn test_rotation_merge_empty_context() {
    let pass = RotationMerge;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_rotation_merge_invalidates() {
    let pass = RotationMerge;
    let inv = pass.invalidates();
    assert!(inv.contains(&"analysis"));
    assert!(inv.contains(&"quantum_analysis"));
}

// ═══════════════════════════════════════════════════════════
// FlashAttentionPass tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_flash_attention_name() {
    let pass = FlashAttentionPass::default();
    assert_eq!(pass.name(), "flash-attention");
}

#[test]
fn test_flash_attention_default_threshold() {
    let pass = FlashAttentionPass::default();
    assert_eq!(pass.seq_len_threshold, 512);
}

#[test]
fn test_flash_attention_custom_threshold() {
    let pass = FlashAttentionPass { seq_len_threshold: 1024 };
    assert_eq!(pass.seq_len_threshold, 1024);
}

#[test]
fn test_flash_attention_empty_context() {
    let pass = FlashAttentionPass::default();
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_flash_attention_invalidates() {
    let pass = FlashAttentionPass::default();
    assert!(pass.invalidates().contains(&"analysis"));
}

// ═══════════════════════════════════════════════════════════
// CommonSubexprElimination tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_cse_name() {
    let pass = CommonSubexprElimination;
    assert_eq!(pass.name(), "common-subexpr-elimination");
}

#[test]
fn test_cse_empty_context() {
    let pass = CommonSubexprElimination;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_cse_invalidates() {
    let pass = CommonSubexprElimination;
    assert!(pass.invalidates().contains(&"analysis"));
}

// ═══════════════════════════════════════════════════════════
// QuantisationPass tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_quantisation_name() {
    let pass = QuantisationPass::default();
    assert_eq!(pass.name(), "quantisation");
}

#[test]
fn test_quantisation_default() {
    let pass = QuantisationPass::default();
    assert_eq!(pass.target_dtype, QuantTarget::Int8);
    assert_eq!(pass.mode, QuantMode::Dynamic);
}

#[test]
fn test_quantisation_int4() {
    let pass = QuantisationPass {
        target_dtype: QuantTarget::Int4,
        mode: QuantMode::Static,
    };
    assert_eq!(pass.target_dtype, QuantTarget::Int4);
    assert_eq!(pass.mode, QuantMode::Static);
}

#[test]
fn test_quantisation_fp8() {
    let pass = QuantisationPass {
        target_dtype: QuantTarget::Fp8E4M3,
        mode: QuantMode::Dynamic,
    };
    assert_eq!(pass.target_dtype, QuantTarget::Fp8E4M3);
}

#[test]
fn test_quantisation_empty_context() {
    let pass = QuantisationPass::default();
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_quantisation_invalidates() {
    let pass = QuantisationPass::default();
    assert!(pass.invalidates().contains(&"analysis"));
}

// ═══════════════════════════════════════════════════════════
// NoiseAwareSchedule tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_noise_aware_schedule_name() {
    let pass = NoiseAwareSchedule;
    assert_eq!(pass.name(), "noise-aware-schedule");
}

#[test]
fn test_noise_aware_schedule_empty_context() {
    let pass = NoiseAwareSchedule;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_noise_aware_schedule_invalidates() {
    let pass = NoiseAwareSchedule;
    assert!(pass.invalidates().contains(&"quantum_analysis"));
}

// ═══════════════════════════════════════════════════════════
// LayoutMapping tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_layout_mapping_name() {
    let pass = LayoutMapping;
    assert_eq!(pass.name(), "layout-mapping");
}

#[test]
fn test_layout_mapping_empty_context() {
    let pass = LayoutMapping;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_layout_mapping_invalidates() {
    let pass = LayoutMapping;
    assert!(pass.invalidates().contains(&"quantum_analysis"));
}

// ═══════════════════════════════════════════════════════════
// Existing passes still work
// ═══════════════════════════════════════════════════════════

#[test]
fn test_dce_on_empty() {
    let pass = lift_opt::DeadCodeElimination;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_constant_fold_on_empty() {
    let pass = lift_opt::ConstantFolding;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_tensor_fusion_on_empty() {
    let pass = lift_opt::TensorFusion;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_gate_cancel_on_empty() {
    let pass = lift_opt::GateCancellation;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

#[test]
fn test_canonicalize_on_empty() {
    let pass = lift_opt::Canonicalize;
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    let result = pass.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Unchanged);
}

// ═══════════════════════════════════════════════════════════
// QuantTarget / QuantMode enum tests
// ═══════════════════════════════════════════════════════════

#[test]
fn test_quant_target_variants() {
    assert_ne!(QuantTarget::Int8, QuantTarget::Int4);
    assert_ne!(QuantTarget::Fp8E4M3, QuantTarget::Fp8E5M2);
    assert_eq!(QuantTarget::Int8, QuantTarget::Int8);
}

#[test]
fn test_quant_mode_variants() {
    assert_ne!(QuantMode::Dynamic, QuantMode::Static);
    assert_eq!(QuantMode::Dynamic, QuantMode::Dynamic);
}
