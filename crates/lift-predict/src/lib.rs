//! LIFT Predict: Performance prediction engine.
//!
//! Provides analytical roofline modelling and budget enforcement for
//! estimating execution time, arithmetic intensity, and bottleneck
//! identification on GPU targets (A100, H100).

pub mod budget;
pub mod roofline;

pub use budget::*;
pub use roofline::*;
