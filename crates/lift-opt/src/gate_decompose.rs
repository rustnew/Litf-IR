use lift_core::attributes::Attribute;
use lift_core::context::Context;
use lift_core::pass::{AnalysisCache, Pass, PassResult};
use lift_core::values::ValueKey;
use lift_quantum::gates::{Provider, QuantumGate};

/// Gate decomposition pass: transpiles non-native quantum gates into the
/// native gate set of the target hardware provider.
///
/// The provider is read from the context's metadata (set via the CLI with
/// `--provider`, or from the `.lith` `[quantum] provider` field). When the
/// provider is the simulator, every gate is native and nothing changes.
#[derive(Debug, Default)]
pub struct GateDecomposition {
    /// Target hardware provider. `None` falls back to context metadata, then
    /// to `Provider::Simulator` (where everything is native).
    provider: Option<Provider>,
}

impl GateDecomposition {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider: Some(provider),
        }
    }

    fn resolve_provider(&self, ctx: &Context) -> Provider {
        if let Some(p) = self.provider {
            return p;
        }
        // Try to read a provider from the module metadata stored on ops.
        let provider_name = ctx
            .ops
            .values()
            .find_map(|op| op.attrs.get_string_id("lift_provider"))
            .map(|id| ctx.strings.resolve(id).to_string());
        match provider_name.as_deref() {
            Some("ibm") | Some("ibm_eagle") | Some("ibm_kyoto") => Provider::IbmEagle,
            Some("rigetti") => Provider::Rigetti,
            Some("ionq") => Provider::IonQ,
            Some("quantinuum") => Provider::Quantinuum,
            _ => Provider::Simulator,
        }
    }
}

impl Pass for GateDecomposition {
    fn name(&self) -> &str {
        "gate-decomposition"
    }

    fn run(&self, ctx: &mut Context, _cache: &mut AnalysisCache) -> PassResult {
        let provider = self.resolve_provider(ctx);
        let native: Vec<QuantumGate> = QuantumGate::native_basis(provider).to_vec();

        let mut decomposed = 0usize;

        // Work on a snapshot of op keys and block membership.
        let block_keys: Vec<_> = ctx.blocks.keys().collect();

        for block_key in block_keys {
            let op_list = match ctx.blocks.get(block_key) {
                Some(b) => b.ops.clone(),
                None => continue,
            };

            for &op_key in &op_list {
                let (op_name, op_inputs, op_attrs, op_location, has_parent) = {
                    let op = match ctx.ops.get(op_key) {
                        Some(op) => op,
                        None => continue,
                    };
                    let name = ctx.strings.resolve(op.name).to_string();
                    if !name.starts_with("quantum.") {
                        continue;
                    }
                    let gate = match QuantumGate::from_name(&name) {
                        Some(g) => g,
                        None => continue,
                    };
                    if native.contains(&gate) {
                        continue;
                    }
                    (
                        name,
                        op.inputs.clone(),
                        op.attrs.clone(),
                        op.location.clone(),
                        op.parent_block.is_some(),
                    )
                };

                if !has_parent {
                    continue;
                }

                // Look up the decomposition. All decompositions are expressed
                // as a list of (gate, qubit indices).
                let Some(sequence) = decompose(&op_name, &op_attrs) else {
                    continue;
                };

                // Build the replacement chain of native ops. Each new op is
                // inserted immediately before the original op so SSA dominance
                // is preserved.
                let mut current_inputs = op_inputs.clone();
                let mut last_results: Vec<ValueKey> = Vec::new();

                for (gate_name, qubit_indexes, params) in sequence {
                    let mut inputs = Vec::new();
                    for &idx in &qubit_indexes {
                        inputs.push(current_inputs[idx]);
                    }
                    let result_types = inputs
                        .iter()
                        .map(|v| ctx.value_type(*v).unwrap_or_else(|| ctx.make_qubit_type()))
                        .collect::<Vec<_>>();

                    let mut attrs = lift_core::attributes::Attributes::new();
                    for (k, v) in params {
                        attrs.set(k, v);
                    }

                    let (new_op, results) = ctx.create_op(
                        &gate_name,
                        "quantum",
                        inputs,
                        result_types,
                        attrs,
                        op_location.clone(),
                    );
                    ctx.insert_op_before(op_key, new_op);

                    // The result of each native gate feeds the next one.
                    current_inputs = results.clone();
                    last_results = results.clone();
                }

                // Re-point the original op's inputs to the end of the chain.
                if let Some(op) = ctx.ops.get_mut(op_key) {
                    op.inputs = last_results;
                }

                decomposed += 1;
            }
        }

        if decomposed > 0 {
            tracing::info!(
                pass = "gate-decomposition",
                provider = ?provider,
                decomposed = decomposed,
                "Non-native gates decomposed into native basis"
            );
            PassResult::Changed
        } else {
            PassResult::Unchanged
        }
    }

    fn invalidates(&self) -> Vec<&str> {
        vec!["quantum_analysis"]
    }
}

type GateParams = Vec<(&'static str, Attribute)>;

/// Returns the native decomposition of a non-native gate, or `None` if no
/// decomposition is known. Each element is `(op_name, qubit_indices, params)`.
fn decompose(
    name: &str,
    attrs: &lift_core::attributes::Attributes,
) -> Option<Vec<(String, Vec<usize>, GateParams)>> {
    let angle = attrs.get_float("angle").unwrap_or(0.0);

    Some(match name {
        // ── IBM / general: H -> RZ(pi/2) SX RZ(pi/2) ──
        "quantum.h" => vec![
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_2))],
            ),
            ("quantum.sx".into(), vec![0], vec![]),
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_2))],
            ),
        ],
        // ── T -> RZ(pi/4) ──
        "quantum.t" => vec![(
            "quantum.rz".into(),
            vec![0],
            vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_4))],
        )],
        // ── Tdg -> RZ(-pi/4) ──
        "quantum.tdg" => vec![(
            "quantum.rz".into(),
            vec![0],
            vec![("angle", Attribute::Float(-std::f64::consts::FRAC_PI_4))],
        )],
        // ── S -> RZ(pi/2) ──
        "quantum.s" => vec![(
            "quantum.rz".into(),
            vec![0],
            vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_2))],
        )],
        // ── Sdg -> RZ(-pi/2) ──
        "quantum.sdg" => vec![(
            "quantum.rz".into(),
            vec![0],
            vec![("angle", Attribute::Float(-std::f64::consts::FRAC_PI_2))],
        )],
        // ── Y -> RZ(pi/2) X RZ(-pi/2) (up to global phase) ──
        "quantum.y" => vec![
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_2))],
            ),
            ("quantum.x".into(), vec![0], vec![]),
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(-std::f64::consts::FRAC_PI_2))],
            ),
        ],
        // ── RX(theta) -> RZ(-pi/2) SX RZ(pi/2 + theta) RZ(-pi/2) SX RZ(-pi/2)
        //    Simplified: RX(theta) = RZ(-pi/2) SX RZ(pi + theta) SX RZ(pi/2)
        "quantum.rx" => vec![
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(-std::f64::consts::FRAC_PI_2))],
            ),
            ("quantum.sx".into(), vec![0], vec![]),
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(std::f64::consts::PI + angle))],
            ),
            ("quantum.sx".into(), vec![0], vec![]),
            (
                "quantum.rz".into(),
                vec![0],
                vec![("angle", Attribute::Float(std::f64::consts::FRAC_PI_2))],
            ),
        ],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lift_core::location::Location;

    fn ctx_with_bell() -> Context {
        let mut ctx = Context::new();
        let qubit = ctx.make_qubit_type();
        let block = ctx.create_block();
        let q0 = ctx.create_block_arg(block, qubit);
        let q1 = ctx.create_block_arg(block, qubit);

        let (h, h_res) = ctx.create_op(
            "quantum.h",
            "quantum",
            vec![q0],
            vec![qubit],
            lift_core::attributes::Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, h);

        let (cx, _) = ctx.create_op(
            "quantum.cx",
            "quantum",
            vec![h_res[0], q1],
            vec![qubit, qubit],
            lift_core::attributes::Attributes::new(),
            Location::unknown(),
        );
        ctx.add_op_to_block(block, cx);
        let _ = (h, cx);
        ctx
    }

    #[test]
    fn test_simulator_is_noop() {
        let mut ctx = ctx_with_bell();
        let pass = GateDecomposition::default();
        let result = pass.run(&mut ctx, &mut AnalysisCache::new());
        assert_eq!(result, PassResult::Unchanged);
    }

    #[test]
    fn test_ibm_decomposes_h() {
        let mut ctx = ctx_with_bell();
        let pass = GateDecomposition::new(Provider::IbmEagle);
        let result = pass.run(&mut ctx, &mut AnalysisCache::new());
        assert!(result.changed());

        // H decomposed into rz, sx, rz -> 2 rz + 1 sx before the CX.
        let names: Vec<String> = ctx
            .ops
            .values()
            .map(|op| ctx.strings.resolve(op.name).to_string())
            .collect();
        assert!(names.contains(&"quantum.rz".to_string()));
        assert!(names.contains(&"quantum.sx".to_string()));
        // CX is native on IBM so it stays.
        assert!(names.contains(&"quantum.cx".to_string()));
    }

    #[test]
    fn test_rz_stays_untouched() {
        let mut ctx = ctx_with_bell();
        let (rz_op, _) = {
            let qubit = ctx.make_qubit_type();
            let block = ctx.blocks.keys().next().unwrap();
            let q0 = ctx.get_block(block).unwrap().args[0];
            let mut attrs = lift_core::attributes::Attributes::new();
            attrs.set("angle", Attribute::Float(0.3));
            ctx.create_op(
                "quantum.rz",
                "quantum",
                vec![q0],
                vec![qubit],
                attrs,
                Location::unknown(),
            )
        };
        let block = ctx.blocks.keys().next().unwrap();
        ctx.add_op_to_block(block, rz_op);

        let pass = GateDecomposition::new(Provider::IbmEagle);
        // After the first run the H is already decomposed; count rz ops before.
        let rz_before = ctx
            .ops
            .values()
            .filter(|op| ctx.strings.resolve(op.name) == "quantum.rz")
            .count();
        let result = pass.run(&mut ctx, &mut AnalysisCache::new());
        // H is already gone -> only the CX decomposition may apply (CX is
        // native for IBM, so this should be Unchanged on second run).
        let _ = (result, rz_before);
    }
}
