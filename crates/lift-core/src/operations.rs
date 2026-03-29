use slotmap::new_key_type;
use serde::{Serialize, Deserialize};
use crate::interning::StringId;
use crate::values::ValueKey;
use crate::regions::RegionKey;
use crate::attributes::Attributes;
use crate::location::Location;

new_key_type! {
    pub struct OpKey;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationData {
    pub name: StringId,
    pub dialect: StringId,
    pub inputs: Vec<ValueKey>,
    pub results: Vec<ValueKey>,
    pub attrs: Attributes,
    pub regions: Vec<RegionKey>,
    pub location: Location,
    pub parent_block: Option<crate::blocks::BlockKey>,
}

impl OperationData {
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn num_results(&self) -> usize {
        self.results.len()
    }

    pub fn has_regions(&self) -> bool {
        !self.regions.is_empty()
    }
}
