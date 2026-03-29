use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeInternId(pub u32);

#[derive(Debug, Default)]
pub struct StringInterner {
    map: HashMap<String, StringId>,
    strings: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }

    pub fn resolve(&self, id: StringId) -> &str {
        &self.strings[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct TypeInterner {
    map: HashMap<crate::types::CoreType, TypeInternId>,
    types: Vec<crate::types::CoreType>,
}

impl TypeInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, ty: crate::types::CoreType) -> crate::types::TypeId {
        if let Some(&id) = self.map.get(&ty) {
            return crate::types::TypeId(id);
        }
        let id = TypeInternId(self.types.len() as u32);
        self.types.push(ty.clone());
        self.map.insert(ty, id);
        crate::types::TypeId(id)
    }

    pub fn resolve(&self, id: crate::types::TypeId) -> &crate::types::CoreType {
        &self.types[id.0 .0 as usize]
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}
