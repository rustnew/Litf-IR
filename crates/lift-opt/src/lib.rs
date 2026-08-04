//! LIFT Opt: Optimisation passes for the LIFT compiler framework.
//!
//! Provides 11 passes: canonicalisation, constant folding, dead code
//! elimination, tensor fusion, common subexpression elimination,
//! flash attention replacement, quantisation annotation, gate cancellation,
//! rotation merge, noise-aware scheduling, and layout mapping.

pub mod canonicalize;
pub mod common_subexpr;
pub mod constant_fold;
pub mod dce;
pub mod flash_attention;
pub mod gate_cancel;
pub mod gate_decompose;
pub mod layout_mapping;
pub mod noise_aware_schedule;
pub mod quantisation_pass;
pub mod real_routing;
pub mod rotation_merge;
pub mod tensor_fusion;

pub use canonicalize::Canonicalize;
pub use common_subexpr::CommonSubexprElimination;
pub use constant_fold::ConstantFolding;
pub use dce::DeadCodeElimination;
pub use flash_attention::FlashAttentionPass;
pub use gate_cancel::GateCancellation;
pub use gate_decompose::GateDecomposition;
pub use layout_mapping::LayoutMapping;
pub use noise_aware_schedule::NoiseAwareSchedule;
pub use quantisation_pass::QuantisationPass;
pub use real_routing::RealRouting;
pub use rotation_merge::RotationMerge;
pub use tensor_fusion::TensorFusion;
