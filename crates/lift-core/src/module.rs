use serde::{Serialize, Deserialize};
use crate::interning::StringId;
use crate::functions::FunctionData;
use crate::location::Location;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleData {
    pub name: StringId,
    pub functions: Vec<FunctionData>,
    pub dialects: Vec<StringId>,
    pub location: Location,
}

impl ModuleData {
    pub fn new(name: StringId) -> Self {
        Self {
            name,
            functions: Vec::new(),
            dialects: Vec::new(),
            location: Location::unknown(),
        }
    }

    pub fn add_function(&mut self, func: FunctionData) {
        self.functions.push(func);
    }

    pub fn num_functions(&self) -> usize {
        self.functions.len()
    }

    pub fn find_function(&self, name: StringId) -> Option<&FunctionData> {
        self.functions.iter().find(|f| f.name == name)
    }
}
