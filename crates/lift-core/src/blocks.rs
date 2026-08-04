use crate::operations::OpKey;
use crate::regions::RegionKey;
use crate::values::ValueKey;
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

new_key_type! {
    pub struct BlockKey;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockData {
    pub args: Vec<ValueKey>,
    pub ops: Vec<OpKey>,
    pub parent_region: Option<RegionKey>,
}

impl BlockData {
    pub fn new() -> Self {
        Self {
            args: Vec::new(),
            ops: Vec::new(),
            parent_region: None,
        }
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

impl Default for BlockData {
    fn default() -> Self {
        Self::new()
    }
}
