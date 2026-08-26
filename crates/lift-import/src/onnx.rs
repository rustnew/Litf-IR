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
    pub fn new() -> Self {
        Self
    }

    pub fn import_from_json(
        &self,
        _ctx: &mut Context,
        json: &serde_json::Value,
    ) -> Result<(), OnnxImportError> {
        let graph = json
            .get("graph")
            .ok_or_else(|| OnnxImportError::General("Missing 'graph' field".into()))?;

        let _nodes = graph
            .get("node")
            .and_then(|n| n.as_array())
            .ok_or_else(|| OnnxImportError::General("Missing 'node' array".into()))?;

        // No node-to-op translation is implemented yet — only top-level
        // structure ('graph'/'node') is validated. Returning `Ok(())` with
        // an empty module+function here used to silently discard every node
        // in the graph, so a caller checking only the `Result` would
        // believe the model imported successfully when the IR is actually
        // empty. Fail loudly instead until real translation exists.
        Err(OnnxImportError::General(
            "ONNX node-to-op translation is not implemented yet; only top-level graph structure is validated".into(),
        ))
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: importing a well-formed graph used to silently
    /// succeed with an empty module, discarding every node — a caller
    /// checking only `Result` would believe the model imported when the IR
    /// is actually empty. It must now fail loudly instead.
    #[test]
    fn test_import_fails_loudly_instead_of_silently_discarding_nodes() {
        let mut ctx = Context::new();
        let json = serde_json::json!({
            "graph": {
                "name": "test",
                "node": [{"opType": "Relu", "input": ["x"], "output": ["y"]}]
            }
        });
        let result = OnnxImporter::new().import_from_json(&mut ctx, &json);
        assert!(
            result.is_err(),
            "unimplemented node translation must not report success"
        );
    }
}
