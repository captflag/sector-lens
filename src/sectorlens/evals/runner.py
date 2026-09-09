"""Run the evaluation cases and score them.

    python -m sectorlens.evals.runner                 # full run, human report
    python -m sectorlens.evals.runner --case pe_buyout_targets_logistics
    python -m sectorlens.evals.runner --json report.json --fail-under 0.7

Runs against whichever provider is configured. Scores are only comparable
within a provider: the deterministic provider composes prose from templates and
will lose most of the persona-concept points by construction, which is the
point -- the gap between the two is the value the language model is adding, and
it is now a number instead of an impression.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from ..agent.core import Agent
from ..agent.schemas import AgentRequest
from ..config import get_persona, load_personas
from ..db.store import connect
from ..settings import get_settings
from .rubric import (CaseResult, Universe, discipline_checks, divergence_check,
                     grounding_checks, persona_checks)

CASES_PATH = Path(__file__).resolve().parents[3] / "evals" / "cases.yaml"

DIMENSIONS = ("grounding", "discipline", "persona")


def load_cases(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or CASES_PATH).read_text())


async def run_case(agent: Agent, case: dict[str, Any], persona: str,
                   rubrics: dict[str, Any], universe: Universe) -> CaseResult:
    response = await agent.ask(AgentRequest(
        query=case["query"], persona=persona, sector=case["sector"]))
    expects = case.get("expects", {}) or {}

    result = CaseResult(case_id=case["id"], persona=persona,
                        sector=case["sector"], query=case["query"],
                        response=response)
    result.checks.extend(grounding_checks(response, expects, universe))
    result.checks.extend(discipline_checks(response, expects))
    result.checks.extend(persona_checks(
        response, rubrics.get(persona, []),
        answer_sections=get_persona(persona).answer_sections))
    return result


async def run_all(agent: Agent, spec: dict[str, Any],
                  only: str | None = None) -> list[CaseResult]:
    settings = get_settings()
    with connect(settings.resolved_db_path, read_only=True) as conn:
        universe = Universe(conn)

    rubrics = spec["persona_rubrics"]
    all_personas = list(load_personas())
    results: list[CaseResult] = []

    for case in spec["cases"]:
        if only and case["id"] != only:
            continue
        personas = case.get("personas", ["pe_analyst"])
        if personas == ["all"]:
            personas = all_personas

        group: list[CaseResult] = []
        for persona in personas:
            group.append(await run_case(agent, case, persona, rubrics, universe))

        if case.get("expects", {}).get("divergent_across_personas"):
            check = divergence_check(group)
            # Attach to the first result so the case owns the finding once.
            group[0].checks.append(check)

        results.extend(group)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 18) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def print_report(results: list[CaseResult], provider: str) -> float:
    print()
    print("=" * 78)
    print(f"Sector Lens evaluation  --  provider: {provider}  --  "
          f"{len(results)} case runs")
    print("=" * 78)

    for result in results:
        status = "FAIL" if result.has_critical_failure else (
            "PASS" if result.score >= 0.8 else "WEAK")
        print(f"\n[{status}] {result.case_id}  ({result.persona})  "
              f"score {result.score:.2f}  {_bar(result.score)}")
        for check in result.failed:
            marker = {"critical": "!!", "major": " *", "minor": "  "}[check.severity]
            print(f"   {marker} {check.id}: {check.detail}")

    print("\n" + "-" * 78)
    print("By dimension")
    for dimension in DIMENSIONS:
        scores = [s for s in (r.dimension_score(dimension) for r in results)
                  if s is not None]
        if scores:
            mean = sum(scores) / len(scores)
            print(f"  {dimension:<12} {mean:.2f}  {_bar(mean)}")

    overall = sum(r.score for r in results) / len(results) if results else 0.0
    criticals = sum(1 for r in results if r.has_critical_failure)
    print("-" * 78)
    print(f"  {'OVERALL':<12} {overall:.2f}  {_bar(overall)}")
    if criticals:
        print(f"\n  {criticals} case run(s) had a critical failure and scored 0. "
              f"Grounding failures are disqualifying by design.")
    print()
    return overall


def to_json(results: list[CaseResult], provider: str, overall: float) -> dict:
    return {
        "provider": provider,
        "overall": round(overall, 4),
        "dimensions": {
            d: round(
                sum(s for s in (r.dimension_score(d) for r in results)
                    if s is not None)
                / max(1, len([1 for r in results if r.dimension_score(d) is not None])),
                4)
            for d in DIMENSIONS
        },
        "cases": [
            {
                "case_id": r.case_id,
                "persona": r.persona,
                "sector": r.sector,
                "score": round(r.score, 4),
                "critical_failure": r.has_critical_failure,
                "confidence": r.response.confidence,
                "companies_referenced": r.response.companies_referenced,
                "tool_calls": [c.tool for c in r.response.tool_calls],
                "checks": [asdict(c) for c in r.checks],
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", help="run a single case by id")
    parser.add_argument("--json", type=Path, help="also write a JSON report here")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="exit non-zero if the overall score is below this")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.resolved_db_path.exists():
        print("No database. Build it first: python -m sectorlens.ingest.build_db",
              file=sys.stderr)
        return 2

    spec = load_cases()
    agent = Agent()
    results = asyncio.run(run_all(agent, spec, only=args.case))
    if not results:
        print(f"No cases matched {args.case!r}", file=sys.stderr)
        return 2

    provider = settings.effective_provider()
    overall = print_report(results, provider)

    if args.json:
        args.json.write_text(json.dumps(to_json(results, provider, overall), indent=2))
        print(f"JSON report written to {args.json}")

    if args.fail_under is not None and overall < args.fail_under:
        print(f"FAILED: overall {overall:.2f} is below the "
              f"--fail-under threshold of {args.fail_under:.2f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
