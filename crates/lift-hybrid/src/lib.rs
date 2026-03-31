//! LIFT Hybrid: Classical-quantum fusion dialect for the LIFT compiler framework.
//!
//! Provides 21 hybrid operations for variational quantum circuits (VQC, VQE,
//! QAOA), gradient methods (parameter shift, adjoint differentiation, SPSA),
//! data encoding strategies, GPU↔QPU transfers, and co-execution policies.

pub mod encoding;
pub mod ops;
pub mod gradient;
pub mod dialect;

pub use encoding::*;
pub use ops::*;
pub use gradient::*;
pub use dialect::HybridDialect;
