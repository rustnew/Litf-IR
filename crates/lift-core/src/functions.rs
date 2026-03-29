use serde::{Serialize, Deserialize};
use crate::interning::StringId;
use crate::types::TypeId;
use crate::regions::RegionKey;
use crate::location::Location;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionData {
    pub name: StringId,
    pub params: Vec<TypeId>,
    pub returns: Vec<TypeId>,
    pub body: Option<RegionKey>,
    pub location: Location,
    pub is_declaration: bool,
}

impl FunctionData {
    pub fn new(name: StringId, params: Vec<TypeId>, returns: Vec<TypeId>) -> Self {
        Self {
            name,
            params,
            returns,
            body: None,
            location: Location::unknown(),
            is_declaration: false,
        }
    }

    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    pub fn num_returns(&self) -> usize {
        self.returns.len()
    }
}
