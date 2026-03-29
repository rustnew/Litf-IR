use lift_sim::cost::Budget;
use lift_sim::analysis::AnalysisReport;
use crate::roofline::RooflineResult;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetCheck {
    pub passed: bool,
    pub violations: Vec<String>,
}

pub fn check_budget(
    report: &AnalysisReport,
    prediction: &RooflineResult,
    budget: &Budget,
) -> BudgetCheck {
    let mut violations = Vec::new();

    if let Err(e) = budget.check_flops(report.total_flops) {
        violations.push(e);
    }
    if let Err(e) = budget.check_memory(report.peak_memory_bytes) {
        violations.push(e);
    }
    if let Some(max_time) = budget.max_time_ms {
        if prediction.predicted_time_ms > max_time {
            violations.push(format!(
                "Time budget exceeded: {:.2}ms > {:.2}ms",
                prediction.predicted_time_ms, max_time
            ));
        }
    }

    BudgetCheck {
        passed: violations.is_empty(),
        violations,
    }
}
