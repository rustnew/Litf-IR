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
    pub fn new() -> Self {
        Self
    }

    pub fn import_from_source(
        &self,
        _ctx: &mut Context,
        source: &str,
    ) -> Result<(), QasmImportError> {
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

        // No gate/register parsing is implemented yet — only the version
        // header above is checked. Returning `Ok(())` with an empty
        // module+function here used to silently discard every gate in
        // `source`, so a caller checking only the `Result` would believe the
        // circuit imported successfully when the IR is actually empty.
        // Fail loudly instead until real parsing exists.
        Err(QasmImportError::General(
            "QASM gate/register parsing is not implemented yet; only the version header is validated".into(),
        ))
    }
}

impl Default for OpenQasm3Importer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: importing a well-formed header used to silently
    /// succeed with an empty module, discarding every gate — a caller
    /// checking only `Result` would believe the circuit imported when the
    /// IR is actually empty. It must now fail loudly instead.
    #[test]
    fn test_import_fails_loudly_instead_of_silently_discarding_gates() {
        let mut ctx = Context::new();
        let source = "OPENQASM 3;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n";
        let result = OpenQasm3Importer::new().import_from_source(&mut ctx, source);
        assert!(
            result.is_err(),
            "unimplemented gate parsing must not report success"
        );
    }
}
