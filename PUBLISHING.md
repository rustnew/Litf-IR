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
> [`.github/workflows/publish.yml`](.github/workflows/publish.yml), which
> publishes every crate in dependency order. See
> [CONTRIBUTING.md](CONTRIBUTING.md#publishing).

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

Ready-to-post texts for each channel are in [docs/ANNOUNCEMENTS.md](docs/ANNOUNCEMENTS.md).

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