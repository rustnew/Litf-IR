use slotmap::new_key_type;
use serde::{Serialize, Deserialize};
use crate::types::TypeId;
use crate::interning::StringId;

new_key_type! {
    pub struct ValueKey;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueData {
    pub ty: TypeId,
    pub name: Option<StringId>,
    pub def: DefSite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefSite {
    OpResult { op: crate::operations::OpKey, result_index: u32 },
    BlockArg { block: crate::blocks::BlockKey, arg_index: u32 },
}
