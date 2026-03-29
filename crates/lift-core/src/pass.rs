use std::collections::HashMap;
use crate::context::Context;

#[derive(Debug, Clone, PartialEq)]
pub enum PassResult {
    Unchanged,
    Changed,
    RolledBack,
    Error(String),
}

impl PassResult {
    pub fn changed(&self) -> bool {
        matches!(self, PassResult::Changed)
    }

    pub fn rolled_back() -> Self {
        PassResult::RolledBack
    }
}

pub trait Pass: std::fmt::Debug {
    fn name(&self) -> &str;
    fn run(&self, ctx: &mut Context, cache: &mut AnalysisCache) -> PassResult;
    fn invalidates(&self) -> Vec<&str> {
        Vec::new()
    }
}

#[derive(Debug, Default)]
pub struct AnalysisCache {
    entries: HashMap<String, Box<dyn std::any::Any>>,
}

impl AnalysisCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: 'static>(&mut self, key: &str, value: T) {
        self.entries.insert(key.to_string(), Box::new(value));
    }

    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.entries.get(key)?.downcast_ref::<T>()
    }

    pub fn invalidate(&mut self, keys: Vec<&str>) {
        for key in keys {
            self.entries.remove(key);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Debug)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    cache: AnalysisCache,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            cache: AnalysisCache::new(),
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run_all(&mut self, ctx: &mut Context) -> Vec<(String, PassResult)> {
        let mut results = Vec::new();

        for pass in &self.passes {
            let name = pass.name().to_string();
            let snapshot = ctx.snapshot();
            let result = pass.run(ctx, &mut self.cache);

            if result.changed() {
                self.cache.invalidate(pass.invalidates());
            }

            results.push((name, result));
            let _ = snapshot;
        }

        results
    }

    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct NoOpPass;

    impl Pass for NoOpPass {
        fn name(&self) -> &str { "no-op" }
        fn run(&self, _ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
            PassResult::Unchanged
        }
    }

    #[test]
    fn test_pass_manager_empty() {
        let mut pm = PassManager::new();
        let mut ctx = Context::new();
        let results = pm.run_all(&mut ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pass_manager_noop() {
        let mut pm = PassManager::new();
        pm.add_pass(Box::new(NoOpPass));
        let mut ctx = Context::new();
        let results = pm.run_all(&mut ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, PassResult::Unchanged);
    }
}
