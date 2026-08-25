# LIFT — Publishing & Visibility Tracker

This document tracks where LIFT is published and referenced across the Rust,
AI, and quantum ecosystems, plus the channels still to pursue.

## Published & live

| Channel | URL | Status |
|---------|-----|--------|
| crates.io (13 crates) | https://crates.io/crates/lift-core | ✅ v0.4.4 |
| docs.rs (13 crates) | https://docs.rs/lift-core | ✅ |
| GitHub repo | https://github.com/rustnew/Lift | ✅ |
| GitHub Releases | https://github.com/rustnew/Lift/releases | ✅ 7 releases |
| GitHub Pages (docs book) | https://rustnew.github.io/Lift/ | ✅ |
| GitHub Discussions | https://github.com/rustnew/Lift/discussions | ✅ |
| crates.io Trusted Publishing | 13 crates → `rustnew/Lift` workflow `publish.yml` | ✅ configured |

> **Publishing is now secure**: all 13 crates use [Trusted Publishing](https://crates.io/docs/trusted-publishing)
> (OIDC, no API token). Pushing a `v*` tag triggers
> [`.github/workflows/publish.yml`](../.github/workflows/publish.yml), which
> publishes every crate in dependency order. See
> [CONTRIBUTING.md](../CONTRIBUTING.md#publishing).

## Pull requests submitted (awaiting merge)

| List | PR | Section |
|------|----|---------|
| qosf/awesome-quantum-software | [#178](https://github.com/qosf/awesome-quantum-software/pull/178) | Quantum full-stack libraries + Quantum compilers (Rust) |
| merrymercy/awesome-tensor-compilers | [#47](https://github.com/merrymercy/awesome-tensor-compilers/pull/47) | Open Source Projects |
| rust-unofficial/awesome-rust | [#2689](https://github.com/rust-unofficial/awesome-rust/pull/2689) | Machine learning |

## To do — other awesome lists

| List | Section | Status |
|------|---------|--------|
| invictvs-choi/awesome-quantum-compiler | — | Skipped — list is research-papers only, not open-source tools |
| zwang4/awesome-machine-learning-in-compilers | — | Skipped — list is "ML applied to compilers", not "ML compilers" |

## Community announcements

Ready-to-post texts for each channel are in [ANNOUNCEMENTS.md](ANNOUNCEMENTS.md).

| Channel | Status |
|---------|--------|
| This Week in Rust | Text ready — submit via https://this-week-in-rust.org/ |
| users.rust-lang.org (Announcements) | Text ready |
| Reddit r/rust | Text ready |
| Reddit r/QuantumComputing | Text ready |
| Reddit r/MachineLearning | Text ready |
| Hacker News (Show HN) | Text ready |
| Lobste.rs | Reuse the HN/r/rust text |
| Rust Discord / Zulip | Reuse the announcement text |

## To do — academic / long-term

| Channel | When | Notes |
|---------|------|-------|
| arXiv paper | v1.0 (Q4 2026) | Already in roadmap |
| Papers With Code | After arXiv | Link the repo |
| Quantum Open Source Foundation (QOSF) | Any time | Community + mentorship |
| Unitary Fund | Any time | Grants for open-source quantum projects |

## Notes

- The crate name `lift` is taken on crates.io (a DB migration tool, unrelated).
  `lift-ir` is available if a standalone brand name is ever needed.
- lib.rs indexes crates.io automatically; no manual submission needed.

## Capabilities & readiness (v0.4.4) — truth check for marketing & reprise

> Written 2026-08-05 (pause until ~2026-09). Keep this in sync with every
> release so the messaging never overpromises. **Rule of thumb**: announce what
> the *code* does today, not what the roadmap plans.

### ✅ Already usable today

| Area | Status | Audience |
|------|--------|----------|
| IR construction (modules, functions, blocks, ops, regions, values, types) | Solid | Compiler developers |
| Dialects: 110 tensor ops, 48 quantum gates, 21 hybrid ops (full types/API) | Real | API consumers |
| 13 optimisation passes (fusion, DCE, rewrites…) + pass framework | Real | Pass developers |
| IR verifier | Real | Program validation |
| Quantum analysis: circuit depth, estimated fidelity, depolarising noise | Real but **static** | Estimation only, no execution |
| Export: ONNX / QASM / LLVM-IR text | Partial | Prototyping |

### ❌ NOT yet usable (honest gaps — these are the v0.5/v0.6 plan)

| Gap | Impact |
|-----|--------|
| No real simulator — `quantum_sim.rs` is static analysis, not state-vector simulation | Cannot run a circuit to get states/amplitudes |
| Importers are ~55-line skeletons (ONNX / PyTorch FX / QASM), not full parsers | Cannot load a real `.onnx` / `.qasm` file end-to-end |
| No real LLVM lowering — backend emits IR text, not executable bytecode | Cannot compile-and-run natively |
| No tensor execution — numpy-like interpreter is planned (v0.5) | Tensor ops do not compute yet |

### 🎯 One-line positioning (use in all marketing)

> **"Rust compiler framework: unified SSA IR for AI + quantum, 13 optimisation
> passes, O0-O3 pipelines, LLVM/ONNX/QASM backends."**

This is accurate today. It is a **framework** (build compilers with it), not yet
an end-to-end compiler you can feed a model/circuit into and run.
