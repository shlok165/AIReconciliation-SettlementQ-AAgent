"""Export reproducible reconciliation reports from a completed pipeline run."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from app.engine.reconcile import ReconciliationResult
from app.evaluation.metrics import EvaluationMetrics


def generate_final_report(
    result: ReconciliationResult, evaluation: EvaluationMetrics, *, output_dir: Path,
) -> Dict[str, str]:
    """Write a machine-readable summary and a complete exception CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exceptions_path = output_dir / "exceptions.csv"
    summary_path = output_dir / "reconciliation_report.json"
    exception_rows = [asdict(exception) for exception in result.exceptions]
    pd.DataFrame(exception_rows, columns=["exception_type", "record_id", "record_type", "description", "related_ids"]).to_csv(exceptions_path, index=False)
    summary = {
        "totals": asdict(result.metrics),
        "evaluation": evaluation.as_dict(),
        "resolution_stage_breakdown": {
            "deterministic_confirmed_matches": result.metrics.deterministic_confirmed_matches,
            "fuzzy_auto_matches": result.metrics.fuzzy_auto_matches,
            "manual_review_candidates": result.metrics.manual_review_candidates,
            "rejected_fuzzy_candidates": result.metrics.rejected_fuzzy_candidates,
        },
        "exception_count": len(result.exceptions),
        "exception_report": str(exceptions_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": str(summary_path), "exceptions": str(exceptions_path)}
