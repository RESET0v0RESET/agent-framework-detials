#!/usr/bin/env python3
"""Detect BP1 memory-management evidence in LangChain/LangGraph repositories.

The detector is intentionally restricted to user-confirmed official APIs/classes
and documented flows. Aggregation is repository-level: each repository can count
at most once for Trim, Delete, and Summary respectively.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rq3_filters import is_mock_like_name, iter_py_files as rq3_iter_py_files


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
}

NON_IMPL_DIR_PARTS = {
    "test",
    "tests",
    "testing",
    "unit_tests",
    "integration_tests",
    "e2e",
    "examples",
    "example",
    "docs",
    "doc",
    "documentation",
    "tutorials",
    "tutorial",
    "cookbook",
    "cookbooks",
    "demo",
    "demos",
    "benchmark",
    "benchmarks",
    ".ipynb_checkpoints",
}

OFFICIAL_ROOTS = ("langchain", "langchain_core", "langchain_classic", "langgraph", "langmem")

TRIM_APIS = {
    "trim_messages": "trim_messages",
    "ConversationBufferWindowMemory": "ConversationBufferWindowMemory",
}

DELETE_APIS = {
    "RemoveMessage": "RemoveMessage",
}

SUMMARY_APIS = {
    "SummarizationMiddleware": "SummarizationMiddleware",
    "SummarizationNode": "SummarizationNode",
    "ConversationSummaryMemory": "ConversationSummaryMemory",
    "ConversationSummaryBufferMemory": "ConversationSummaryBufferMemory",
}

SUMMARY_SUPPORT_NAMES = {
    "RunningSummary",
}

EXCLUDED_APIS = {
    "MessagesPlaceholder",
    "ConversationTokenBufferMemory",
    "ContextEditingMiddleware",
    "ClearToolUsesEdit",
}


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


def is_non_impl_path(path: Path, repo_dir: Path) -> bool:
    rel_parts = {part.lower() for part in path.relative_to(repo_dir).parts[:-1]}
    if rel_parts & NON_IMPL_DIR_PARTS:
        return True
    name = path.name.lower()
    return (
        name == "conftest.py"
        or name.startswith(("test_", "tests_", "example_", "examples_", "demo_", "tutorial_", "benchmark_"))
        or name.endswith(("_test.py", "_tests.py", "_example.py", "_demo.py", "_tutorial.py", "_benchmark.py"))
    )


def iter_py_files(repo_dir: Path, exclude_non_impl_paths: bool = False) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        if exclude_non_impl_paths:
            dirs[:] = [d for d in dirs if d.lower() not in NON_IMPL_DIR_PARTS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            if exclude_non_impl_paths and is_non_impl_path(path, repo_dir):
                continue
            yield path


iter_py_files = rq3_iter_py_files


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
    return bool(module) and module.startswith(OFFICIAL_ROOTS)


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, file_path: Path, rel_path: str, source: str):
        self.framework = framework
        self.repo = repo
        self.file_path = file_path
        self.rel_path = rel_path
        self.source = source
        self.lines = source.splitlines()
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.evidence: list[Evidence] = []
        self.running_summary_seen = False
        self.summarization_node_seen = False

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
                self.official_module_aliases.add(alias.asname or alias.name.split(".")[0])
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

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in SUMMARY_SUPPORT_NAMES and node.id in self.alias_to_qualified:
            self.running_summary_seen = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        confirmed = self.is_confirmed_official_call(full, last)

        if confirmed and last in TRIM_APIS:
            self.add("Trim", TRIM_APIS[last], node)

        if confirmed and last in DELETE_APIS:
            rule = "RemoveMessage"
            if self.call_has_remove_all_messages(node):
                rule = "RemoveMessage_REMOVE_ALL_MESSAGES"
            self.add("Delete", rule, node)

        if confirmed and last in SUMMARY_APIS:
            if last == "SummarizationNode":
                self.summarization_node_seen = True
            else:
                self.add("Summary", SUMMARY_APIS[last], node)

        if confirmed and last in SUMMARY_SUPPORT_NAMES:
            self.running_summary_seen = True

        self.generic_visit(node)

    def call_has_remove_all_messages(self, node: ast.Call) -> bool:
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Name) and arg.id == "REMOVE_ALL_MESSAGES":
                return True
            if isinstance(arg, ast.Constant) and arg.value == "REMOVE_ALL_MESSAGES":
                return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        if self.is_custom_summary_flow(node):
            self.add("Summary", "custom_summary_flow_with_RemoveMessage", node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        if self.is_custom_summary_flow(node):
            self.add("Summary", "custom_summary_flow_with_RemoveMessage", node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def is_custom_summary_flow(self, node: ast.AST) -> bool:
        function_name = getattr(node, "name", "").lower()
        if "summar" not in function_name:
            return False

        has_remove_message = False
        has_model_invoke = False
        returns_summary = False

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                full, last = call_name(child.func)
                if last == "RemoveMessage" and self.is_confirmed_official_call(full, last):
                    has_remove_message = True
                if isinstance(child.func, ast.Attribute) and child.func.attr in {"invoke", "ainvoke"}:
                    has_model_invoke = True
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                for key in child.value.keys:
                    if isinstance(key, ast.Constant) and key.value == "summary":
                        returns_summary = True

        return has_remove_message and has_model_invoke and returns_summary

    def finalize(self) -> list[Evidence]:
        if self.summarization_node_seen and self.running_summary_seen:
            # Use the first line with SummarizationNode as representative evidence.
            for ev in self.evidence:
                if ev.rule == "SummarizationNode":
                    return self.evidence
            match = re.search(r"\bSummarizationNode\s*\(", self.source)
            lineno = self.source[: match.start()].count("\n") + 1 if match else 0
            self.evidence.append(
                Evidence(
                    framework=self.framework,
                    repo=self.repo,
                    category="Summary",
                    rule="SummarizationNode_with_RunningSummary",
                    file=self.rel_path,
                    line=lineno,
                    code=self.line_text(lineno),
                )
            )
        return self.evidence


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

    rel_path = str(path.relative_to(repo_dir))
    analyzer = FileAnalyzer(framework, repo, path, rel_path, source)
    analyzer.visit(tree)
    return analyzer.finalize()


def analyze_file_by_confirmed_regex(
    framework: str, repo: str, repo_dir: Path, path: Path, source: str
) -> list[Evidence]:
    """Fallback for syntax-invalid files. Match only confirmed API call forms."""
    rel_path = str(path.relative_to(repo_dir))
    patterns = [
        ("Trim", "trim_messages", r"\btrim_messages\s*\("),
        ("Trim", "ConversationBufferWindowMemory", r"\bConversationBufferWindowMemory\s*\("),
        ("Delete", "RemoveMessage", r"\bRemoveMessage\s*\("),
        ("Summary", "SummarizationMiddleware", r"\bSummarizationMiddleware\s*\("),
        ("Summary", "ConversationSummaryMemory", r"\bConversationSummaryMemory\s*\("),
        ("Summary", "ConversationSummaryBufferMemory", r"\bConversationSummaryBufferMemory\s*\("),
    ]
    evidence: list[Evidence] = []
    lines = source.splitlines()
    for category, rule, pattern in patterns:
        m = re.search(pattern, source)
        if not m:
            continue
        lineno = source[: m.start()].count("\n") + 1
        code = lines[lineno - 1].strip()[:300] if 1 <= lineno <= len(lines) else ""
        evidence.append(Evidence(framework, repo, category, rule, rel_path, lineno, code))
    if re.search(r"\bSummarizationNode\s*\(", source) and re.search(r"\bRunningSummary\b", source):
        m = re.search(r"\bSummarizationNode\s*\(", source)
        lineno = source[: m.start()].count("\n") + 1 if m else 0
        code = lines[lineno - 1].strip()[:300] if 1 <= lineno <= len(lines) else ""
        evidence.append(Evidence(framework, repo, "Summary", "SummarizationNode_with_RunningSummary", rel_path, lineno, code))
    return evidence


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    evidence: list[Evidence] = []
    py_count = 0
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            evidence.extend(analyze_file(framework, repo_name, repo_dir, path))
        except Exception:
            continue

    categories = {ev.category for ev in evidence}
    has_trim = "Trim" in categories
    has_delete = "Delete" in categories
    has_summary = "Summary" in categories
    defect = (has_trim or has_delete) and not has_summary
    rules_by_category: dict[str, set[str]] = {"Trim": set(), "Delete": set(), "Summary": set()}
    for ev in evidence:
        rules_by_category.setdefault(ev.category, set()).add(ev.rule)

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "has_trim": has_trim,
        "has_delete": has_delete,
        "has_summary": has_summary,
        "bp1_defect": defect,
        "trim_rules": ";".join(sorted(rules_by_category.get("Trim", set()))),
        "delete_rules": ";".join(sorted(rules_by_category.get("Delete", set()))),
        "summary_rules": ";".join(sorted(rules_by_category.get("Summary", set()))),
        "evidence": evidence,
    }


def list_repos(base: Path, frameworks: list[str]) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name):
            if entry.is_dir():
                repos.append((framework, entry.name, str(entry)))
    return repos


def scale_count(count: int, population: int, target: int) -> int:
    if population <= 0:
        return 0
    return int((count * target / population) + 0.5)


def target_sample_size(population: int) -> int:
    twenty = int((population * 0.2) + 0.5)
    return twenty if twenty > 200 else 200


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_csv = out_dir / "bp1_repo_results.csv"
    evidence_csv = out_dir / "bp1_evidence.csv"
    summary_csv = out_dir / "bp1_summary_by_framework.csv"
    scaled_csv = out_dir / "bp1_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "has_trim",
                "has_delete",
                "has_summary",
                "bp1_defect",
                "trim_rules",
                "delete_rules",
                "summary_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "category", "rule", "file", "line", "code"],
        )
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

    frameworks = sorted({row["framework"] for row in results})
    summary_rows = []
    scaled_rows = []
    for fw in frameworks:
        rows = [row for row in results if row["framework"] == fw]
        population = len(rows)
        trim = sum(1 for row in rows if row["has_trim"])
        delete = sum(1 for row in rows if row["has_delete"])
        summary = sum(1 for row in rows if row["has_summary"])
        defect = sum(1 for row in rows if row["bp1_defect"])
        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_trim": trim,
                "repos_with_delete": delete,
                "repos_with_summary": summary,
                "bp1_defect_repos": defect,
            }
        )
        target = target_sample_size(population)
        scaled_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "target_report_repos": target,
                "scaled_repos_with_trim": scale_count(trim, population, target),
                "scaled_repos_with_delete": scale_count(delete, population, target),
                "scaled_repos_with_summary": scale_count(summary, population, target),
                "scaled_bp1_defect_repos": scale_count(defect, population, target),
            }
        )

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp1_memory")
    parser.add_argument("--framework", action="append", default=["langchain", "langgraph"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument(
        "--exclude-non-impl-paths",
        action="store_true",
        help="Exclude test/example/documentation/demo/benchmark paths from analysis",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    repos = list_repos(base, args.framework)
    if args.limit:
        limited: list[tuple[str, str, str]] = []
        for fw in args.framework:
            fw_repos = [r for r in repos if r[0] == fw][: args.limit]
            limited.extend(fw_repos)
        repos = limited

    print(f"Analyzing {len(repos)} repositories: {', '.join(args.framework)}", file=sys.stderr)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(results, out_dir)
    print(f"Wrote results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
