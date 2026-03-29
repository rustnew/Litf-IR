use serde::{Serialize, Deserialize};
use crate::interning::StringId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub file: Option<StringId>,
    pub line: u32,
    pub column: u32,
}

impl Location {
    pub fn unknown() -> Self {
        Self { file: None, line: 0, column: 0 }
    }

    pub fn new(file: StringId, line: u32, column: u32) -> Self {
        Self { file: Some(file), line, column }
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(_file) = self.file {
            write!(f, "<file>:{}:{}", self.line, self.column)
        } else {
            write!(f, "<unknown>:{}:{}", self.line, self.column)
        }
    }
}
