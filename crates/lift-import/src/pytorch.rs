use lift_core::context::Context;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PyTorchImportError {
    #[error("Unknown PyTorch op: {0}")]
    UnknownOp(String),
    #[error("Import error: {0}")]
    General(String),
}

#[derive(Debug)]
pub struct PyTorchFxImporter;

impl PyTorchFxImporter {
    pub fn new() -> Self { Self }

    pub fn import_from_json(&self, ctx: &mut Context, json: &serde_json::Value) -> Result<(), PyTorchImportError> {
        let nodes = json.get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| PyTorchImportError::General("Missing 'nodes' array".into()))?;

        let module_idx = ctx.create_module("pytorch_import");
        let func_name = ctx.intern_string("forward");
        let func = lift_core::functions::FunctionData::new(func_name, vec![], vec![]);
        ctx.add_function_to_module(module_idx, func);

        let _ = nodes.len();
        Ok(())
    }
}

impl Default for PyTorchFxImporter {
    fn default() -> Self { Self::new() }
}
