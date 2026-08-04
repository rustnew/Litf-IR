use crate::interning::StringId;
use crate::types::TypeId;
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

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
    OpResult {
        op: crate::operations::OpKey,
        result_index: u32,
    },
    BlockArg {
        block: crate::blocks::BlockKey,
        arg_index: u32,
    },
}
