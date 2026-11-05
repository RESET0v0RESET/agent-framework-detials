#!/usr/bin/env python3
"""Create the framework-stratified RQ3 sample repo list."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


FRAMEWORK_ORDER = [
    "langchain",
    "langgraph",
    "crewai",
    "llamaindex",
    "autogen",
    "haystack",
    "semantic_kernel",
    "camel",
    "autogpt",
    "metagpt",
]

PAPER_POPULATION = {
    "langchain": 29387,
    "langgraph": 16517,
    "crewai": 8687,
    "llamaindex": 8301,
    "autogen": 4208,
    "haystack": 3687,
    "semantic_kernel": 1428,
    "camel": 353,
    "autogpt": 228,
    "metagpt": 218,
}


def norm_framework(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def report_target(population: int) -> int:
    twenty = int(population * 0.2 + 0.5)
    return twenty if twenty > 200 else 200


def load_repos(summary_csv: Path) -> dict[str, list[str]]:
    repos: dict[str, list[str]] = {fw: [] for fw in FRAMEWORK_ORDER}
    seen: set[tuple[str, str]] = set()
    with summary_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            framework = norm_framework(str(row.get("dataset_framework") or row.get("framework") or ""))
            repo = str(row.get("repo") or "").strip()
            if framework not in repos or not repo:
                continue
            key = (framework, repo)
            if key in seen:
                continue
            seen.add(key)
            repos[framework].append(repo)
    for framework in repos:
        repos[framework].sort()
    return repos


def write_outputs(repos: dict[str, list[str]], out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "rq3_stratified_sample_repos.csv"
    summary_path = out_dir / "rq3_stratified_sample_summary.csv"
    rng = random.Random(seed)

    summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    for framework in FRAMEWORK_ORDER:
        all_repos = repos[framework]
        population = PAPER_POPULATION[framework]
        target = report_target(population)
        physical_n = min(target, len(all_repos))
        if physical_n < len(all_repos):
            selected = sorted(rng.sample(all_repos, physical_n))
        else:
            selected = list(all_repos)
        scaling_factor = target / physical_n if physical_n else 0.0
        summary_rows.append(
            {
                "dataset_framework": framework,
                "paper_population": population,
                "report_target": target,
                "server_repos": len(all_repos),
                "physical_sample_repos": physical_n,
                "scaling_factor": f"{scaling_factor:.8f}",
                "seed": seed,
            }
        )
        for idx, repo in enumerate(selected, 1):
            sample_rows.append(
                {
                    "dataset_framework": framework,
                    "repo": repo,
                    "sample_index": idx,
                    "paper_population": population,
                    "report_target": target,
                    "physical_sample_repos": physical_n,
                    "scaling_factor": f"{scaling_factor:.8f}",
                    "seed": seed,
                }
            )

    with sample_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset_framework",
            "repo",
            "sample_index",
            "paper_population",
            "report_target",
            "physical_sample_repos",
            "scaling_factor",
            "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset_framework",
            "paper_population",
            "report_target",
            "server_repos",
            "physical_sample_repos",
            "scaling_factor",
            "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(sample_path)
    print(summary_path)
    print(f"sample_rows {len(sample_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-summary", default="/home/zhaiyibo/zyb/rq3_static_repro/results/agent_tool_descriptions_precise_v1_full/repo_description_summary.csv")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/rq3_stratified_sample_v1")
    parser.add_argument("--seed", type=int, default=20260625)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repos = load_repos(Path(args.repo_summary))
    write_outputs(repos, Path(args.out_dir), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
