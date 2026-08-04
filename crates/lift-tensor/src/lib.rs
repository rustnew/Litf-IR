//! LIFT Tensor: AI dialect for the LIFT compiler framework.
//!
//! Provides 90+ tensor operations covering arithmetic, activations,
//! normalisation, attention variants, convolutions, pooling, quantisation,
//! diffusion, GNN, parallelism, fused ops, and gradient operations.
//! Includes shape inference, FLOP counting, and memory estimation.

pub mod dialect;
pub mod ops;
pub mod shape;
pub mod types;

pub use dialect::TensorDialect;
pub use ops::*;
pub use shape::ShapeInference;
pub use types::*;
