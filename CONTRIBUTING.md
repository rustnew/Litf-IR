# Contributing to LIFT

Thanks for your interest in contributing to LIFT — a unified intermediate
representation for AI and quantum computing.

This guide covers the development workflow, project layout, and how to get
your changes reviewed and merged.

## Table of contents

- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Building and testing](#building-and-testing)
- [Code style](#code-style)
- [Validation](#validation)
- [Publishing](#publishing)
- [Commit conventions](#commit-conventions)
- [Opening a pull request](#opening-a-pull-request)

## Development setup

Requirements:

- **Rust** 1.80 or newer (see `rust-version` in `Cargo.toml`)
- **Cargo** (comes with Rust)

Clone and build:

```bash
git clone git@github.com:rustnew/Lift.git
cd Lift
cargo build --workspace
```

## Project layout

LIFT is a Cargo workspace of 13 published crates, organised by dependency layer:

| Layer | Crates | Purpose |
|-------|--------|---------|
| L0 — Foundation | `lift-core`, `lift-config` | SSA IR, verifier; O0–O3 pipeline config |
| L1 — Dialects & Frontend | `lift-ast`, `lift-tensor`, `lift-quantum` | lexer/parser; AI ops; quantum gates & noise |
| L2 — Analysis & I/O | `lift-opt`, `lift-sim`, `lift-export`, `lift-import`, `lift-hybrid` | passes; cost model; backends; importers; fusion |
| L3 — Prediction | `lift-predict` | roofline / performance prediction |
| L4 — Tools | `lift-cli`, `lift-codegen` | CLI; programmatic model generation |

`lift-tests` (`publish = false`) holds the integration test suite. `lift-demo`
(`publish = false`) is a standalone, end-to-end hybrid AI+quantum pipeline
walkthrough — useful as a worked example, not part of the library API.

## Building and testing

```bash
# Build the whole workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Build a single crate
cargo build -p lift-core
```

## Code style

- Run `rustfmt` — CI enforces `cargo fmt --all --check`.
- Run `clippy` with warnings denied — CI enforces `cargo clippy --all-targets -- -D warnings`.
- Keep changes minimal and focused on a single concern.

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

## Validation

Before submitting, run the end-to-end validation script, which exercises the
full pipeline (`verify → analyse → optimise → predict → export`) across all
example models:

```bash
bash examples/validate_all.sh
```

This is also run in CI on every push to `main` and on pull requests.

## Publishing

Releases are published to [crates.io](https://crates.io). The process:

1. Bump the version in `Cargo.toml` (`[workspace.package] version`) and update
   version references across `README.md` and the docs (`docs/LIFT_Guide.md`,
   `docs/LIFT_Manual.md`, `docs/LIFT_design.md`, `docs/DIALECTS.md`).
2. Update `CHANGELOG.md`.
3. Push a version tag — the [`publish` workflow](.github/workflows/publish.yml)
   publishes all 13 crates automatically in dependency order
   (L0 → L1 → L2 → L3 → L4) using **Trusted Publishing** (OIDC, no API token):

   ```bash
   git tag v0.4.6
   git push origin v0.4.6
   ```

   > Trusted Publishing is configured per crate on crates.io (Settings →
   > Trusted Publishing) for `rustnew/Lift`, workflow `publish.yml`. The
   > workflow can also be triggered manually via the Actions tab
   > (`workflow_dispatch`).
   >
   > crates.io does not allow overwriting a published version — a fix to an
   > already-published release requires a new version bump.

4. Create a GitHub release:

   ```bash
   gh release create v0.4.6 --title "..." --notes "..."
   ```

### Manual fallback

If you need to publish outside CI (e.g. the very first release of a new
crate), publish in dependency order with the API token:

```bash
cargo publish -p lift-core
cargo publish -p lift-config
# ... then L1, L2, L3, L4 ...
```

## Commit conventions

Use conventional commit prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `chore:` — maintenance (bumps, metadata, tooling)
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests

Example: `docs: add vision, roadmap, and layer-graph diagrams to README`

## Opening a pull request

1. Fork the repository and create a feature branch.
2. Make your changes, keeping them focused.
3. Run `cargo fmt`, `cargo clippy`, `cargo test`, and `bash examples/validate_all.sh`.
4. Push your branch and open a pull request against `main`.
5. CI runs automatically (fmt, clippy, tests, validation). All checks must pass
   before merge.

Thank you for contributing to LIFT!