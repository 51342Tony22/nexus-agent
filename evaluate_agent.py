"""CLI for the explicit-cost v1.8 paper-analysis Agent evaluation."""

import argparse
import json
import tempfile
from pathlib import Path

from agent_evaluator import (
    PROJECT_DIR,
    load_evaluation_cases,
    run_live_evaluation,
)


def print_case_manifest(cases: list[dict]) -> None:
    print("Paper Analysis Evaluation Cases")
    print("=" * 72)
    for index, case in enumerate(cases, start=1):
        exists = (PROJECT_DIR / "files" / case["paper"]).is_file()
        print(
            f"{index}. {case['case_id']} | paper={case['paper']} "
            f"| file_exists={exists}"
        )
        print(f"   goal={case['evaluation_goal']}")


def format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def print_live_report(report: dict) -> None:
    print("\nLive Agent Evaluation")
    print("=" * 72)
    for item in report["case_results"]:
        print(
            f"{item['case_id']}: status={item['structured_status']} "
            f"repair={item['repair_attempted']} "
            f"citations={item['citation_count']} "
            f"prompt_est_tokens={item['prompt_estimated_tokens']} "
            f"tokens={item['total_tokens']} "
            f"latency={item['latency_seconds']}"
        )
        if item.get("error"):
            print(
                "  error="
                + json.dumps(item["error"], ensure_ascii=False)
            )

    aggregate = report["aggregate"]
    print("\nAggregate")
    print(f"Cases completed: {aggregate['completed_count']}/{aggregate['case_count']}")
    print(
        "Structured valid: "
        f"{format_rate(aggregate['structured_valid_rate'])}"
    )
    print(
        "First-pass valid: "
        f"{format_rate(aggregate['first_pass_valid_rate'])}"
    )
    print(
        "Repair attempted: "
        f"{format_rate(aggregate['repair_attempt_rate'])}"
    )
    print(
        "Repair success (attempted cases): "
        f"{format_rate(aggregate['repair_success_rate'])}"
    )
    print(
        "Citation identity valid (valid outputs): "
        f"{format_rate(aggregate['citation_identity_valid_rate'])}"
    )
    print(
        "Retrieval sufficient: "
        f"{format_rate(aggregate['retrieval_sufficient_rate'])}"
    )
    print(f"Model calls: {aggregate['total_model_calls']}")
    print(f"Total tokens: {aggregate['total_tokens']}")
    print(f"Average tokens: {aggregate['average_tokens']}")
    print(f"Total latency: {aggregate['total_latency_seconds']:.2f}s")
    print(f"Average latency: {aggregate['average_latency_seconds']}")
    print(f"\nScope: {report['metric_scope']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect six local evaluation cases without API calls, or run "
            "them explicitly with --live."
        )
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured model for every selected case.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases (useful for a one-case smoke test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for a live report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_evaluation_cases()
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than zero.")
        cases = cases[:args.limit]

    print_case_manifest(cases)
    if not args.live:
        print("\nDry run only: no embedding model or LLM API was called.")
        print("Use '--live --limit 1' for a one-paper smoke test.")
        return

    from agent import FYPResearchAgent

    with tempfile.TemporaryDirectory() as directory:
        agent = FYPResearchAgent(
            memory_store_path=str(Path(directory) / "evaluation_memory.json")
        )
        report = run_live_evaluation(cases=cases, agent=agent)

    print_live_report(report)
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
