// ============================================================================
// report.rs — Test Harness and Formatting Utilities
// ============================================================================
//
// Provides:
//   - `TestReport`: accumulates PASS/FAIL results across all pipeline steps.
//   - `check()`: records a single assertion and prints its outcome.
//   - `print_step()`: prints a step header for visual pipeline separation.
//   - `format_flops()` / `format_bytes()`: human-readable numeric formatting.
//
// ============================================================================

/// Accumulates test results across the entire pipeline run.
#[derive(Debug, Default)]
pub struct TestReport {
    pub passed: u32,
    pub failed: u32,
}

impl TestReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single test assertion. Prints `[PASS]` or `[FAIL]` to stdout.
    pub fn check(&mut self, name: &str, ok: bool) {
        if ok {
            println!("    [PASS] {}", name);
            self.passed += 1;
        } else {
            println!("    [FAIL] {}", name);
            self.failed += 1;
        }
    }

    /// Total number of assertions recorded.
    pub fn total(&self) -> u32 {
        self.passed + self.failed
    }

    /// Returns `true` if every recorded assertion passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Print the final summary box to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  FINAL REPORT                                              ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║  Passed: {:>3}                                               ║", self.passed);
        println!("║  Failed: {:>3}                                               ║", self.failed);
        println!("║  Total:  {:>3}                                               ║", self.total());
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

/// Print a pipeline step header.
pub fn print_step(n: u32, title: &str) {
    println!();
    println!("── Step {} ── {}", n, title);
}

/// Format a FLOP count into a human-readable string (FLOP / KFLOP / MFLOP / GFLOP / TFLOP).
pub fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} TFLOP", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} GFLOP", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} MFLOP", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.2} KFLOP", flops as f64 / 1e3)
    } else {
        format!("{} FLOP", flops)
    }
}

/// Format a byte count into a human-readable string (B / KiB / MiB / GiB).
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1_024 {
        format!("{:.2} KiB", bytes as f64 / 1_024.0)
    } else {
        format!("{} B", bytes)
    }
}
