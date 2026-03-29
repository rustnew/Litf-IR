use slotmap::new_key_type;
use serde::{Serialize, Deserialize};
use crate::operations::OpKey;
use crate::values::ValueKey;
use crate::regions::RegionKey;

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
