#!/usr/bin/env python
"""
evals/run_evals.py — Repeatable evaluation CLI

Usage:
    # Run all cases (recommended entry point)
    python -m evals.run_evals

    # Run specific case IDs
    python -m evals.run_evals --cases AGENT-001 TOOL-002 RAG-003

    # Run by tag
    python -m evals.run_evals --tag rag

    # Run by type (agent | rag | tool)
    python -m evals.run_evals --type agent

    # Fail fast (exit 1 on first failure)
    python -m evals.run_evals --fail-fast

    # Quiet mode (suppress verbose progress, still writes report)
    python -m evals.run_evals --quiet

    # Custom report directory
    python -m evals.run_evals --report-dir /tmp/eval_reports

    # CI mode: exit 1 if overall score below threshold
    python -m evals.run_evals --min-score 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_evals",
        description="Agentic Platform — AI Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="CASE_ID",
        help="Run only specific case IDs (e.g. AGENT-001 RAG-003)",
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        help="Run only cases with this tag (e.g. rag, calculator, robustness)",
    )
    parser.add_argument(
        "--type",
        choices=["agent", "rag", "tool"],
        dest="eval_type",
        metavar="TYPE",
        help="Run only cases of this type: agent | rag | tool",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (report is still written)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop execution after the first failed case",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        metavar="SCORE",
        help="Exit 1 if overall score is below this threshold (0.0–1.0). Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--report-dir",
        metavar="PATH",
        help="Directory to write the JSON report. Default: evals/reports/",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    from evals.dataset.cases import (
        EVAL_DATASET,
        EvalType,
        get_case_by_id,
        get_cases_by_tag,
        get_cases_by_type,
    )
    from evals.runner import run_evaluation

    # ── Case selection ─────────────────────────────────────────────────────
    cases = EVAL_DATASET

    if args.cases:
        selected = []
        for cid in args.cases:
            case = get_case_by_id(cid)
            if case is None:
                print(f"ERROR: Unknown case ID '{cid}'", file=sys.stderr)
                return 2
            selected.append(case)
        cases = selected

    elif args.tag:
        cases = get_cases_by_tag(args.tag)
        if not cases:
            print(f"ERROR: No cases found with tag '{args.tag}'", file=sys.stderr)
            return 2

    elif args.eval_type:
        target = EvalType(args.eval_type)
        cases = get_cases_by_type(target)
        if not cases:
            print(f"ERROR: No cases found with type '{args.eval_type}'", file=sys.stderr)
            return 2

    # ── Fail-fast wrapper ──────────────────────────────────────────────────
    if args.fail_fast:
        # Run cases one by one; abort on first failure
        from evals.runner import run_evaluation

        for case in cases:
            partial_report = await run_evaluation(
                cases=[case],
                verbose=not args.quiet,
                report_dir=args.report_dir,
            )
            if partial_report.failed > 0:
                print(f"\nFail-fast: stopping after failure in {case.id}.", file=sys.stderr)
                return 1
        return 0

    # ── Full run ───────────────────────────────────────────────────────────
    report = await run_evaluation(
        cases=cases,
        verbose=not args.quiet,
        report_dir=args.report_dir,
    )

    # ── CI exit code ──────────────────────────────────────────────────────
    if report.failed > 0:
        return 1
    if args.min_score > 0 and report.overall_score < args.min_score:
        print(
            f"\nCI threshold not met: score={report.overall_score:.2%} < min={args.min_score:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
