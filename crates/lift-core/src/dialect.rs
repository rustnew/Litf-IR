use std::collections::HashMap;
pub trait Dialect: std::fmt::Debug {
    fn name(&self) -> &str;
    fn verify_op(&self, op_name: &str, num_inputs: usize, num_results: usize) -> Result<(), String>;
}

#[derive(Debug, Default)]
pub struct DialectRegistry {
    dialects: HashMap<String, Box<dyn Dialect>>,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, dialect: Box<dyn Dialect>) {
        self.dialects.insert(dialect.name().to_string(), dialect);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Dialect> {
        self.dialects.get(name).map(|d| d.as_ref())
    }

    pub fn has(&self, name: &str) -> bool {
        self.dialects.contains_key(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.dialects.keys().map(|s| s.as_str()).collect()
    }
}

#[derive(Debug)]
pub struct CoreDialect;

impl Dialect for CoreDialect {
    fn name(&self) -> &str { "core" }

    fn verify_op(&self, op_name: &str, _num_inputs: usize, _num_results: usize) -> Result<(), String> {
        match op_name {
            "core.constant" | "core.return" | "core.call" | "core.br" | "core.cond_br" => Ok(()),
            _ => Err(format!("Unknown core operation: {}", op_name)),
        }
    }
}

pub fn register_builtin_dialects(registry: &mut DialectRegistry) {
    registry.register(Box::new(CoreDialect));
}
