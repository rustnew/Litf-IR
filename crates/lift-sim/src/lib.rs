//! LIFT Sim: Static analysis and cost modelling engine.
//!
//! Provides FLOP counting, memory analysis, noise simulation, roofline cost
//! models (A100/H100), quantum cost models (superconducting, trapped-ion,
//! neutral-atom), energy/carbon estimation, and reactive budget tracking.

pub mod analysis;
pub mod cost;
pub mod quantum_sim;

pub use analysis::*;
pub use cost::*;
pub use quantum_sim::*;
