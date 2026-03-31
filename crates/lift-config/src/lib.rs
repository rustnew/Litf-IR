//! LIFT Config: Configuration language parser for `.lith` files.
//!
//! Provides parsing and validation of LIFT compilation configuration,
//! including target backend, budget constraints, optimisation levels,
//! simulation toggles, and quantum device settings.

pub mod parser;
pub mod types;

pub use parser::ConfigParser;
pub use types::*;
