use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::interning::StringId;
use crate::types::TypeId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Integer(i64),
    Float(f64),
    String(StringId),
    Bool(bool),
    Type(TypeId),
    Array(Vec<Attribute>),
    Dict(HashMap<String, Attribute>),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Attributes {
    entries: HashMap<String, Attribute>,
}

impl Attributes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: Attribute) {
        self.entries.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&Attribute> {
        self.entries.get(key)
    }

    pub fn get_integer(&self, key: &str) -> Option<i64> {
        match self.entries.get(key)? {
            Attribute::Integer(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.entries.get(key)? {
            Attribute::Float(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.entries.get(key)? {
            Attribute::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string_id(&self, key: &str) -> Option<StringId> {
        match self.entries.get(key)? {
            Attribute::String(v) => Some(*v),
            _ => None,
        }
    }

    pub fn remove(&mut self, key: &str) -> Option<Attribute> {
        self.entries.remove(key)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Attribute)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl PartialEq for Attributes {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}
