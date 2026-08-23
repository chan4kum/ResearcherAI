r"""
evals/regression.py — LLM Regression Testing and Quality Gate

Evaluates changes against golden baselines and enforced thresholds:
- Verifies overall score >= min_overall_score (default 85%)
- Verifies per-dimension scores >= dimension thresholds
- Detects significant quality regressions vs. baseline (max drop 5%)
- Provides change-detection triggers for CI path filtering
- Generates structured Markdown reports for CI Step Summaries
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.runner import EvalSuiteReport, run_evaluation
from evals.thresholds import DEFAULT_THRESHOLDS, EVAL_TRIGGER_PATTERNS, QualityThresholds


@dataclass
class DimensionComparison:
    dimension: str
    baseline_score: float
    current_score: float
    delta: float
    min_required: float
    passed_threshold: bool
    passed_regression: bool
    status: str


@dataclass
class RegressionReport:
    timestamp: str
    passed: bool
    overall_score: float
    baseline_overall_score: float
    overall_delta: float
    threshold_failures: list[str]
    regression_failures: list[str]
    dimension_comparisons: list[DimensionComparison]
    total_cases: int
    cases_passed: int
    cases_failed: int

    def to_markdown(self) -> str:
        """Render a formatted markdown summary table for CI step summaries."""
        status_icon = "✅ PASS" if self.passed else "❌ REGRESSION DETECTED"
        lines = [
            f"# LLM Quality & Regression Evaluation Report",
            f"",
            f"**Status**: {status_icon}  ",
            f"**Evaluation Timestamp**: `{self.timestamp}`  ",
            f"**Overall Score**: `{self.overall_score:.2%}` (Baseline: `{self.baseline_overall_score:.2%}`, $\\Delta$: `{self.overall_delta:+.2%}`)  ",
            f"**Cases**: {self.cases_passed}/{self.total_cases} passed ({self.cases_failed} failed)",
            f"",
            f"## Per-Dimension Performance",
            f"",
            f"| Dimension | Current | Baseline | Delta | Threshold | Status |",
            f"|:---|:---:|:---:|:---:|:---:|:---:|",
        ]

        for d in self.dimension_comparisons:
            delta_str = f"{d.delta:+.2%}"
            if d.delta > 0:
                delta_str = f"+{d.delta:.2%}"
            lines.append(
                f"| `{d.dimension}` | `{d.current_score:.2%}` | `{d.baseline_score:.2%}` | "
                f"`{delta_str}` | `>= {d.min_required:.2%}` | {d.status} |"
            )

        if self.threshold_failures:
            lines.extend([
                f"",
                f"### ⚠️ Quality Threshold Breaches",
                f"",
            ])
            for f in self.threshold_failures:
                lines.append(f"- ❌ {f}")

        if self.regression_failures:
            lines.extend([
                f"",
                f"### 📉 Significant Regressions vs. Baseline",
                f"",
            ])
            for f in self.regression_failures:
                lines.append(f"- ❌ {f}")

        lines.extend([
            f"",
            f"---",
            f"*Evaluation executed by Agentic Platform Quality Gate (Milestone 49)*",
        ])
        return "\n".join(lines)


def load_baseline_report(baseline_path: str | Path | None = None) -> dict[str, Any]:
    """Load baseline evaluation JSON report."""
    path = Path(baseline_path) if baseline_path else Path(__file__).parent / "baselines" / "baseline.json"
    if not path.is_file():
        raise FileNotFoundError(f"Baseline report not found at: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_paths_trigger_eval(changed_files: list[str]) -> tuple[bool, list[str]]:
    """Check if any changed file matches the defined evaluation trigger patterns."""
    matched: list[str] = []
    for file_path in changed_files:
        clean = file_path.strip().replace("\\", "/")
        for pattern in EVAL_TRIGGER_PATTERNS:
            if fnmatch.fnmatch(clean, pattern) or fnmatch.fnmatch(os.path.basename(clean), pattern):
                matched.append(clean)
                break
    return len(matched) > 0, matched


def compare_with_baseline(
    current_report: EvalSuiteReport,
    baseline_data: dict[str, Any],
    thresholds: QualityThresholds | None = None,
) -> RegressionReport:
    """Compare a newly generated evaluation suite report against the golden baseline."""
    cfg = thresholds or DEFAULT_THRESHOLDS
    base_overall = float(baseline_data.get("overall_score", 1.0))
    base_dims: dict[str, float] = baseline_data.get("per_dimension_scores", {})

    current_overall = current_report.overall_score
    overall_delta = round(current_overall - base_overall, 4)

    threshold_failures: list[str] = []
    regression_failures: list[str] = []
    dim_comparisons: list[DimensionComparison] = []

    # 1. Overall score checks
    if current_overall < cfg.min_overall_score:
        threshold_failures.append(
            f"Overall score {current_overall:.2%} is below minimum threshold of {cfg.min_overall_score:.2%}"
        )

    if overall_delta < -cfg.max_allowed_drop:
        regression_failures.append(
            f"Overall regression drop {abs(overall_delta):.2%} exceeds maximum allowed drop of {cfg.max_allowed_drop:.2%}"
        )

    # 2. Per-dimension checks
    all_dims = sorted(set(list(current_report.per_dimension_scores.keys()) + list(base_dims.keys())))

    for dim in all_dims:
        curr_score = current_report.per_dimension_scores.get(dim, 1.0)
        base_score = base_dims.get(dim, 1.0)
        delta = round(curr_score - base_score, 4)
        min_req = cfg.min_dimension_scores.get(dim, 0.70)

        passed_threshold = curr_score >= min_req
        passed_regression = delta >= -cfg.max_dimension_drop

        if not passed_threshold:
            threshold_failures.append(
                f"Dimension '{dim}' score {curr_score:.2%} below minimum required {min_req:.2%}"
            )
        if not passed_regression:
            regression_failures.append(
                f"Dimension '{dim}' dropped by {abs(delta):.2%} vs baseline (max allowed: {cfg.max_dimension_drop:.2%})"
            )

        status = "✅ PASS" if (passed_threshold and passed_regression) else "❌ FAIL"
        dim_comparisons.append(
            DimensionComparison(
                dimension=dim,
                baseline_score=base_score,
                current_score=curr_score,
                delta=delta,
                min_required=min_req,
                passed_threshold=passed_threshold,
                passed_regression=passed_regression,
                status=status,
            )
        )

    passed = (len(threshold_failures) == 0) and (len(regression_failures) == 0)

    return RegressionReport(
        timestamp=current_report.timestamp,
        passed=passed,
        overall_score=current_overall,
        baseline_overall_score=base_overall,
        overall_delta=overall_delta,
        threshold_failures=threshold_failures,
        regression_failures=regression_failures,
        dimension_comparisons=dim_comparisons,
        total_cases=current_report.total_cases,
        cases_passed=current_report.passed,
        cases_failed=current_report.failed,
    )


# ---------------------------------------------------------------------------
# CLI Command
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.regression",
        description="LLM Regression Testing and Quality Gate",
    )
    parser.add_argument(
        "--baseline",
        metavar="PATH",
        help="Path to baseline evaluation JSON file (default: evals/baselines/baseline.json)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_THRESHOLDS.min_overall_score,
        help=f"Minimum overall score threshold (0.0–1.0). Default: {DEFAULT_THRESHOLDS.min_overall_score}",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=DEFAULT_THRESHOLDS.max_allowed_drop,
        help=f"Maximum allowed overall drop vs baseline (0.0–1.0). Default: {DEFAULT_THRESHOLDS.max_allowed_drop}",
    )
    parser.add_argument(
        "--check-triggers-only",
        nargs="*",
        metavar="FILE",
        help="Check if supplied file paths match evaluation triggers and exit 0 (matched) or 3 (no match)",
    )
    parser.add_argument(
        "--output-markdown",
        metavar="PATH",
        help="Write markdown summary report to file (e.g. for GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=True,
        help="Exit with code 1 if quality thresholds or regression limits are breached (default: True)",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Mode 1: Check trigger paths only
    if args.check_triggers_only is not None:
        matched, triggers = check_paths_trigger_eval(args.check_triggers_only)
        if matched:
            print(f"Trigger match found ({len(triggers)} files):")
            for t in triggers:
                print(f"  ↳ {t}")
            return 0
        else:
            print("No changes match LLM evaluation trigger paths. Evaluation not required.")
            return 3  # Distinct exit code for workflow skipping

    # Mode 2: Run evaluation and compare with baseline
    thresholds = QualityThresholds(
        min_overall_score=args.min_score,
        max_allowed_drop=args.max_drop,
    )

    baseline_data = load_baseline_report(args.baseline)
    current_report = await run_evaluation(verbose=True)
    regression_report = compare_with_baseline(current_report, baseline_data, thresholds)

    md_output = regression_report.to_markdown()

    # Write to file if requested or if GITHUB_STEP_SUMMARY is present
    if args.output_markdown:
        out_path = Path(args.output_markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_output, encoding="utf-8")
        print(f"\nMarkdown summary saved to: {out_path}")

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_path and Path(step_summary_path).parent.exists():
        with open(step_summary_path, "a", encoding="utf-8") as f:
            f.write(f"\n{md_output}\n")

    if not regression_report.passed and args.fail_on_regression:
        print("\n❌ CI QUALITY GATE FAILED: Significant regression or threshold failure detected.", file=sys.stderr)
        return 1

    print("\n✅ CI QUALITY GATE PASSED: All evaluation thresholds and regression limits met.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
