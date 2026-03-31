//! LIFT Tensor: AI dialect for the LIFT compiler framework.
//!
//! Provides 90+ tensor operations covering arithmetic, activations,
//! normalisation, attention variants, convolutions, pooling, quantisation,
//! diffusion, GNN, parallelism, fused ops, and gradient operations.
//! Includes shape inference, FLOP counting, and memory estimation.

pub mod types;
pub mod ops;
pub mod dialect;
pub mod shape;

pub use types::*;
pub use ops::*;
pub use dialect::TensorDialect;
pub use shape::ShapeInference;
