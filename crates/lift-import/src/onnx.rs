use lift_core::context::Context;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxImportError {
    #[error("Unsupported ONNX opset version: {0}")]
    UnsupportedOpset(i64),
    #[error("Unknown ONNX op: {0}")]
    UnknownOp(String),
    #[error("Import error: {0}")]
    General(String),
}

#[derive(Debug)]
pub struct OnnxImporter;

impl OnnxImporter {
    pub fn new() -> Self { Self }

    pub fn import_from_json(&self, ctx: &mut Context, json: &serde_json::Value) -> Result<(), OnnxImportError> {
        let graph = json.get("graph")
            .ok_or_else(|| OnnxImportError::General("Missing 'graph' field".into()))?;

        let module_idx = ctx.create_module("onnx_import");

        let nodes = graph.get("node")
            .and_then(|n| n.as_array())
            .ok_or_else(|| OnnxImportError::General("Missing 'node' array".into()))?;

        let _node_count = nodes.len();

        // Create a function for the graph
        let func_name = graph.get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("main");

        let name_id = ctx.intern_string(func_name);
        let func = lift_core::functions::FunctionData::new(name_id, vec![], vec![]);
        ctx.add_function_to_module(module_idx, func);

        Ok(())
    }
}

impl Default for OnnxImporter {
    fn default() -> Self { Self::new() }
}
