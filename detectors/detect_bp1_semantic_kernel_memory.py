#!/usr/bin/env python3
"""Detect BP1 D1 memory-management evidence in Semantic Kernel Python repos.

Rule:
- Trim/Delete evidence: AST-confirmed calls to ChatHistoryTruncationReducer(...)
  imported from or accessed through semantic_kernel...
- Summary evidence: AST-confirmed calls to ChatHistorySummarizationReducer(...)
  imported from or accessed through semantic_kernel...
- D1 defect: truncation evidence exists and summarization evidence is absent.

Aggregation is repository-level. A repository counts at most once for each
category even if multiple matching calls are found.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rq3_filters import iter_py_files as rq3_iter_py_files


OFFICIAL_ROOT = "semantic_kernel"
FRAMEWORK_DIR = "semantic_kernel"

TRUNCATION_CLASS = "ChatHistoryTruncationReducer"
SUMMARIZATION_CLASS = "ChatHistorySummarizationReducer"


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
    file: str
    line: int
    code: str


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def attr_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return parts


def call_name(node: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = attr_chain(node)
        if parts:
            return ".".join(parts), parts[-1]
    return None, None


def is_official_module(module: str | None) -> bool:
    return bool(module) and (module == OFFICIAL_ROOT or module.startswith(f"{OFFICIAL_ROOT}."))


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = str(path.relative_to(repo_dir))
        self.source = source
        self.lines = source.splitlines()
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, category: str, rule: str, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_official_module(alias.name):
                local = alias.asname or alias.name.split(".")[0]
                self.official_module_aliases.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if is_official_module(module):
            for alias in node.names:
                local = alias.asname or alias.name
                self.alias_to_qualified[local] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def is_confirmed_official_call(self, full: str | None, last: str | None) -> bool:
        if not last:
            return False
        if last in self.alias_to_qualified:
            return True
        if full:
            parts = full.split(".")
            if parts and parts[0] in self.official_module_aliases:
                return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last):
            if last == TRUNCATION_CLASS:
                self.add("Trim/Delete", TRUNCATION_CLASS, node)
            elif last == SUMMARIZATION_CLASS:
                self.add("Summary", SUMMARIZATION_CLASS, node)
        self.generic_visit(node)


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> list[Evidence]:
    source = read_text(path)
    if source is None:
        return []
    if "\x00" in source:
        source = source.replace("\x00", "")
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return []

    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.evidence


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    evidence: list[Evidence] = []
    py_count = 0
    for path in rq3_iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            evidence.extend(analyze_file(framework, repo_name, repo_dir, path))
        except Exception:
            continue

    categories = {ev.category for ev in evidence}
    has_truncation = "Trim/Delete" in categories
    has_summary = "Summary" in categories
    defect = has_truncation and not has_summary
    rules_by_category: dict[str, set[str]] = {"Trim/Delete": set(), "Summary": set()}
    for ev in evidence:
        rules_by_category.setdefault(ev.category, set()).add(ev.rule)

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "has_trim_delete": has_truncation,
        "has_summary": has_summary,
        "bp1_d1_defect": defect,
        "trim_delete_rules": ";".join(sorted(rules_by_category.get("Trim/Delete", set()))),
        "summary_rules": ";".join(sorted(rules_by_category.get("Summary", set()))),
        "evidence": evidence,
    }


def list_repos(base: Path, framework: str) -> list[tuple[str, str, str]]:
    fw_dir = base / framework
    if not fw_dir.is_dir():
        raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
    return [(framework, entry.name, str(entry)) for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name) if entry.is_dir()]


def scale_count(count: int, population: int, target: int) -> int:
    if population <= 0:
        return 0
    return int((count * target / population) + 0.5)


def target_sample_size(population: int) -> int:
    twenty = int((population * 0.2) + 0.5)
    return twenty if twenty > 200 else 200


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_csv = out_dir / "bp1_semantic_kernel_repo_results.csv"
    evidence_csv = out_dir / "bp1_semantic_kernel_evidence.csv"
    summary_csv = out_dir / "bp1_semantic_kernel_summary.csv"
    scaled_csv = out_dir / "bp1_semantic_kernel_scaled_summary.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "has_trim_delete",
                "has_summary",
                "bp1_d1_defect",
                "trim_delete_rules",
                "summary_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["framework", "repo", "category", "rule", "file", "line", "code"])
        writer.writeheader()
        for row in results:
            seen: set[tuple[str, str, str, str, int]] = set()
            for ev in row["evidence"]:
                key = (ev.framework, ev.repo, ev.category, ev.rule, ev.file, ev.line)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    {
                        "framework": ev.framework,
                        "repo": ev.repo,
                        "category": ev.category,
                        "rule": ev.rule,
                        "file": ev.file,
                        "line": ev.line,
                        "code": ev.code,
                    }
                )

    population = len(results)
    trim_delete = sum(1 for row in results if row["has_trim_delete"])
    summary = sum(1 for row in results if row["has_summary"])
    both = sum(1 for row in results if row["has_trim_delete"] and row["has_summary"])
    defect = sum(1 for row in results if row["bp1_d1_defect"])
    target = target_sample_size(population)

    summary_rows = [
        {
            "framework": FRAMEWORK_DIR,
            "population_repos": population,
            "repos_with_trim_delete": trim_delete,
            "repos_with_summary": summary,
            "repos_with_both": both,
            "bp1_d1_defect_repos": defect,
        }
    ]
    scaled_rows = [
        {
            "framework": FRAMEWORK_DIR,
            "population_repos": population,
            "target_report_repos": target,
            "scaled_repos_with_trim_delete": scale_count(trim_delete, population, target),
            "scaled_repos_with_summary": scale_count(summary, population, target),
            "scaled_repos_with_both": scale_count(both, population, target),
            "scaled_bp1_d1_defect_repos": scale_count(defect, population, target),
        }
    ]

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp1_memory_semantic_kernel_python")
    parser.add_argument("--framework", default=FRAMEWORK_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit for pilot runs")
    parser.add_argument("--exclude-non-impl-paths", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    repos = list_repos(base, args.framework)
    if args.limit:
        repos = repos[: args.limit]

    print(f"Analyzing {len(repos)} repositories: {args.framework}", file=sys.stderr)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 250 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(results, out_dir)
    print(f"Wrote results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
