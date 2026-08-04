//! LIFT Core: SSA-based intermediate representation foundation.
//!
//! This crate provides the core data structures and algorithms for the LIFT
//! compiler framework, including types, values, operations, blocks, regions,
//! functions, modules, attributes, a verifier, an IR printer, a pass manager,
//! and a dialect registry.

pub mod attributes;
pub mod blocks;
pub mod context;
pub mod dialect;
pub mod functions;
pub mod interning;
pub mod location;
pub mod model_builder;
pub mod module;
pub mod operations;
pub mod pass;
pub mod printer;
pub mod regions;
pub mod types;
pub mod values;
pub mod verifier;

pub use attributes::{Attribute, Attributes};
pub use blocks::{BlockData, BlockKey};
pub use context::Context;
pub use dialect::{Dialect, DialectRegistry};
pub use functions::FunctionData;
pub use interning::{StringId, StringInterner, TypeInterner};
pub use location::Location;
pub use model_builder::{
    build_lith_config, tensor, tensor_1d, tensor_2d, tensor_3d, tensor_4d, ModelBuilder, ModelType,
};
pub use module::ModuleData;
pub use operations::{OpKey, OperationData};
pub use pass::{AnalysisCache, Pass, PassManager, PassResult};
pub use printer::Printer;
pub use regions::{RegionData, RegionKey};
pub use types::{CoreType, DataType, TypeId};
pub use values::{DefSite, ValueData, ValueKey};
pub use verifier::{Verifier, VerifyError};
