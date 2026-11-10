"""Run the agent over the test cases and write a results file and a Markdown report.

Usage: python evals/run_eval.py --label baseline [--repeats 1] [--only case_id ...]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evals")]

from cases import build_cases, tools_acceptable
from catalog_vision.agent import ask, build_agent
from catalog_vision.config import REPORT_DIR
from catalog_vision.tools import CollectionStore
from langgraph.errors import GraphRecursionError

RATE_LIMIT_WAIT_SECONDS = 30
MAX_RATE_LIMIT_RETRIES = 6
PAUSE_BETWEEN_CASES_SECONDS = 3


def is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or "rate_limit" in str(error)


def run_with_retries(agent, question: str):
    for _ in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return ask(agent, question)
        except Exception as error:
            if not is_rate_limit(error):
                raise
            print(f"  rate limited, waiting {RATE_LIMIT_WAIT_SECONDS}s", flush=True)
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    return ask(agent, question)


def grade(case, agent) -> dict:
    record = {"case_id": case.case_id, "category": case.category, "question": case.question,
              "expected": case.expected}
    try:
        run = run_with_retries(agent, case.question)
    except GraphRecursionError:
        return {**record, "passed": False, "tools_ok": False, "answer_ok": False,
                "answer": "", "error": "exceeded the step limit", "tools_called": [],
                "tokens": 0, "seconds": 0.0}
    except Exception as error:
        return {**record, "passed": False, "tools_ok": False, "answer_ok": False,
                "answer": "", "error": f"{type(error).__name__}: {str(error)[:300]}",
                "tools_called": [], "tokens": 0, "seconds": 0.0}

    tools_ok = tools_acceptable(case, run.tool_calls)
    answer_ok = bool(run.answer) and case.check(run.answer)
    return {**record, "passed": tools_ok and answer_ok, "tools_ok": tools_ok, "answer_ok": answer_ok,
            "answer": run.answer, "error": None, "tools_called": run.tool_calls,
            "tokens": run.total_tokens, "seconds": run.seconds}


def summarise(results: list[dict]) -> dict:
    by_category = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result["passed"])
    return {
        "runs": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in by_category.items()},
        "total_tokens": sum(r["tokens"] for r in results),
        "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2],
    }


def write_report(label: str, summary: dict, results: list[dict]) -> None:
    lines = [f"# Agent evaluation: {label}", "",
             f"Passed {summary['passed']} of {summary['runs']} runs ({summary['pass_rate']:.0%}). "
             f"Tokens used: {summary['total_tokens']:,}. Median answer time: {summary['median_seconds']}s.",
             "", "| Category | Passed |", "|---|---|"]
    lines += [f"| {c} | {v} |" for c, v in summary["by_category"].items()]
    failures = [r for r in results if not r["passed"]]
    lines += ["", "## Failed runs", ""] if failures else ["", "No failed runs.", ""]
    for r in failures:
        tools = ", ".join(c["name"] for c in r["tools_called"]) or "none"
        answer = " ".join((r["error"] or r["answer"]).split())[:400]
        lines += [f"**{r['case_id']}** ({r['category']})", f"- Question: {r['question']}",
                  f"- Expected: {r['expected']}", f"- Tools called: {tools}",
                  f"- Tools ok: {r['tools_ok']}, answer ok: {r['answer_ok']}", f"- Answer: {answer}", ""]
    (REPORT_DIR / f"eval_report_{label}.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    store = CollectionStore()
    agent = build_agent(store)
    cases = [c for c in build_cases(store) if not args.only or c.case_id in args.only]

    results = []
    for repeat in range(args.repeats):
        for case in cases:
            result = grade(case, agent)
            result["repeat"] = repeat
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case.case_id} "
                  f"({result['tokens']} tokens)", flush=True)
            time.sleep(PAUSE_BETWEEN_CASES_SECONDS)

    summary = summarise(results)
    (REPORT_DIR / f"eval_results_{args.label}.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, default=str) + "\n")
    write_report(args.label, summary, results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
