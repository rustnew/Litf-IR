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
    pub fn new() -> Self {
        Self
    }

    pub fn import_from_json(
        &self,
        _ctx: &mut Context,
        json: &serde_json::Value,
    ) -> Result<(), PyTorchImportError> {
        let _nodes = json
            .get("nodes")
            .and_then(|n| n.as_array())
            .ok_or_else(|| PyTorchImportError::General("Missing 'nodes' array".into()))?;

        // No node-to-op translation is implemented yet — only the top-level
        // 'nodes' array is validated. Returning `Ok(())` with an empty
        // module+function here used to silently discard every node in the
        // FX graph, so a caller checking only the `Result` would believe
        // the model imported successfully when the IR is actually empty.
        // Fail loudly instead until real translation exists.
        Err(PyTorchImportError::General(
            "PyTorch FX node-to-op translation is not implemented yet; only the top-level 'nodes' array is validated".into(),
        ))
    }
}

impl Default for PyTorchFxImporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: importing a well-formed FX graph used to silently
    /// succeed with an empty module, discarding every node — a caller
    /// checking only `Result` would believe the model imported when the IR
    /// is actually empty. It must now fail loudly instead.
    #[test]
    fn test_import_fails_loudly_instead_of_silently_discarding_nodes() {
        let mut ctx = Context::new();
        let json = serde_json::json!({
            "nodes": [{"op": "call_function", "target": "relu"}]
        });
        let result = PyTorchFxImporter::new().import_from_json(&mut ctx, &json);
        assert!(
            result.is_err(),
            "unimplemented node translation must not report success"
        );
    }
}
