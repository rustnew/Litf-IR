use slotmap::new_key_type;
use serde::{Serialize, Deserialize};
use crate::blocks::BlockKey;
use crate::operations::OpKey;

new_key_type! {
    pub struct RegionKey;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionData {
    pub blocks: Vec<BlockKey>,
    pub entry_block: Option<BlockKey>,
    pub parent_op: Option<OpKey>,
}

impl RegionData {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            entry_block: None,
            parent_op: None,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl Default for RegionData {
    fn default() -> Self {
        Self::new()
    }
}
