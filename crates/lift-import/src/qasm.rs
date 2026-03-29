use lift_core::context::Context;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QasmImportError {
    #[error("Unsupported QASM version: {0}")]
    UnsupportedVersion(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Import error: {0}")]
    General(String),
}

#[derive(Debug)]
pub struct OpenQasm3Importer;

impl OpenQasm3Importer {
    pub fn new() -> Self { Self }

    pub fn import_from_source(&self, ctx: &mut Context, source: &str) -> Result<(), QasmImportError> {
        let lines: Vec<&str> = source.lines().collect();

        if lines.is_empty() {
            return Err(QasmImportError::General("Empty source".into()));
        }

        // Check version header
        if let Some(first) = lines.first() {
            if !first.contains("OPENQASM 3") && !first.contains("OPENQASM 2") {
                return Err(QasmImportError::UnsupportedVersion(first.to_string()));
            }
        }

        let module_idx = ctx.create_module("qasm_import");
        let func_name = ctx.intern_string("circuit");
        let func = lift_core::functions::FunctionData::new(func_name, vec![], vec![]);
        ctx.add_function_to_module(module_idx, func);

        Ok(())
    }
}

impl Default for OpenQasm3Importer {
    fn default() -> Self { Self::new() }
}
