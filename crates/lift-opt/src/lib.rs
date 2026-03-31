//! LIFT Opt: Optimisation passes for the LIFT compiler framework.
//!
//! Provides 11 passes: canonicalisation, constant folding, dead code
//! elimination, tensor fusion, common subexpression elimination,
//! flash attention replacement, quantisation annotation, gate cancellation,
//! rotation merge, noise-aware scheduling, and layout mapping.

pub mod dce;
pub mod constant_fold;
pub mod tensor_fusion;
pub mod gate_cancel;
pub mod canonicalize;
pub mod rotation_merge;
pub mod flash_attention;
pub mod common_subexpr;
pub mod quantisation_pass;
pub mod noise_aware_schedule;
pub mod layout_mapping;

pub use dce::DeadCodeElimination;
pub use constant_fold::ConstantFolding;
pub use tensor_fusion::TensorFusion;
pub use gate_cancel::GateCancellation;
pub use canonicalize::Canonicalize;
pub use rotation_merge::RotationMerge;
pub use flash_attention::FlashAttentionPass;
pub use common_subexpr::CommonSubexprElimination;
pub use quantisation_pass::QuantisationPass;
pub use noise_aware_schedule::NoiseAwareSchedule;
pub use layout_mapping::LayoutMapping;
