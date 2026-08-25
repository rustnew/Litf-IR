# Security Policy

## Supported Versions

LIFT is pre-1.0 software under active development. Only the most recently
published version on [crates.io](https://crates.io/crates/lift-core) receives
security fixes.

| Version | Supported |
|---------|-----------|
| Latest  | ✅ |
| Older   | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private reporting flow:

1. Go to the [Security tab](https://github.com/rustnew/Lift/security) of this
   repository.
2. Click **"Report a vulnerability"** to open a private security advisory.

This lets us discuss and fix the issue with you before any details are made
public.

If the vulnerability report concerns a dependency rather than LIFT's own
code, please also consider reporting it upstream to the dependency's
maintainers.

## What to Expect

- We aim to acknowledge new reports within a few days.
- We'll work with you to understand the issue, develop a fix, and agree on a
  disclosure timeline.
- Once a fix is released, we'll credit reporters (unless you'd prefer to
  remain anonymous) in the [CHANGELOG](CHANGELOG.md) and the security
  advisory.

## Scope

This policy covers the LIFT compiler framework itself (the crates in this
repository: `lift-core`, `lift-ast`, `lift-tensor`, `lift-quantum`,
`lift-hybrid`, `lift-opt`, `lift-sim`, `lift-predict`, `lift-import`,
`lift-export`, `lift-config`, `lift-cli`, `lift-codegen`). LIFT does not run
untrusted input by design today — it has no execution runtime — so most
realistic security concerns are in the parser/lexer (malformed `.lif`/`.lith`
input) and the exporters (generated code correctness, not memory safety,
since Rust's type system rules out the classic C/C++ vulnerability classes).
