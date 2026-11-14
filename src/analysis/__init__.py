"""Cold-start and sparse demand analysis package (Member 4 — Phase 2)."""
from src.analysis.cold_start_analysis import (
    segment_by_demand_pattern,
    evaluate_short_window_performance,
    analyze_censoring_bias,
    compare_local_vs_global_by_segment,
    document_failure_cases,
    generate_cold_start_report,
)

__all__ = [
    "segment_by_demand_pattern",
    "evaluate_short_window_performance",
    "analyze_censoring_bias",
    "compare_local_vs_global_by_segment",
    "document_failure_cases",
    "generate_cold_start_report",
]
