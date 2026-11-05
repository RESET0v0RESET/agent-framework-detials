#!/usr/bin/env python3
"""Detect BP3 self-recovery evidence in downstream agent repositories.

The detector uses framework-official API names discussed with the user. It is
AST-based, so comments and docstrings are not matched. Aggregation is
repository-level: each repository can count at most once for Retry,
Termination, and Checkpoint respectively.
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

DEFAULT_FRAMEWORKS = [
    "langchain",
    "langgraph",
    "llamaindex",
    "crewai",
    "autogen",
    "haystack",
    "semantic_kernel",
    "camel",
    "metagpt",
    "autogpt",
]

OFFICIAL_ROOTS_BY_FRAMEWORK = {
    "langchain": ("langchain", "langchain_core", "langchain_classic", "langgraph"),
    "langgraph": ("langgraph", "langchain", "langchain_core", "langchain_classic"),
    "llamaindex": ("llama_index", "llama_agents"),
    "crewai": ("crewai",),
    "autogen": ("autogen", "autogen_agentchat", "autogen_core", "autogen_ext"),
    "haystack": ("haystack",),
    "semantic_kernel": ("semantic_kernel",),
    "camel": ("camel",),
    "metagpt": ("metagpt",),
    "autogpt": ("autogpt",),
}

CALL_RULES = {
    "langchain": {
        "Retry": {
            "RunnableRetry",
            "RetryPolicy",
            "ToolRetryMiddleware",
            "ModelRetryMiddleware",
        },
        "Termination": {
            "ModelCallLimitMiddleware",
            "ToolCallLimitMiddleware",
        },
        "Checkpoint": {
            "InMemorySaver",
            "MemorySaver",
            "SqliteSaver",
            "AsyncSqliteSaver",
            "PostgresSaver",
            "AsyncPostgresSaver",
            "MongoDBSaver",
            "AsyncMongoDBSaver",
        },
    },
    "langgraph": {
        "Retry": {
            "RunnableRetry",
            "RetryPolicy",
            "ToolRetryMiddleware",
            "ModelRetryMiddleware",
        },
        "Termination": {
            "ModelCallLimitMiddleware",
            "ToolCallLimitMiddleware",
        },
        "Checkpoint": {
            "InMemorySaver",
            "MemorySaver",
            "SqliteSaver",
            "AsyncSqliteSaver",
            "PostgresSaver",
            "AsyncPostgresSaver",
            "MongoDBSaver",
            "AsyncMongoDBSaver",
        },
    },
    "llamaindex": {
        "Retry": {
            "RetryPolicy",
            "ConstantDelayRetryPolicy",
            "ExponentialBackoffRetryPolicy",
            "retry_policy",
            "stop_after_attempt",
        },
        "Termination": set(),
        "Checkpoint": {"WorkflowCheckpointer"},
    },
    "crewai": {
        "Retry": set(),
        "Termination": set(),
        "Checkpoint": {"persist"},
    },
    "autogen": {
        "Retry": set(),
        "Termination": {
            "MaxMessageTermination",
            "TextMentionTermination",
            "TokenUsageTermination",
            "TimeoutTermination",
            "ExternalTermination",
            "HandoffTermination",
            "SourceMatchTermination",
            "StopMessageTermination",
            "FunctionalTermination",
        },
        "Checkpoint": set(),
    },
    "haystack": {
        "Retry": {"request_with_retry", "async_request_with_retry"},
        "Termination": set(),
        "Checkpoint": set(),
    },
    "semantic_kernel": {
        "Retry": set(),
        "Termination": {
            "TerminationStrategy",
            "DefaultTerminationStrategy",
            "KernelFunctionTerminationStrategy",
            "AggregatorTerminationStrategy",
        },
        "Checkpoint": set(),
    },
    "camel": {
        "Retry": set(),
        "Termination": {"ResponseWordsTerminator", "TokenLimitTerminator"},
        "Checkpoint": set(),
    },
    "metagpt": {
        "Retry": set(),
        "Termination": set(),
        "Checkpoint": set(),
    },
    "autogpt": {
        "Retry": set(),
        "Termination": set(),
        "Checkpoint": set(),
    },
}

METHOD_RULES = {
    "langchain": {
        "Retry": {"with_retry"},
        "Termination": set(),
        "Checkpoint": set(),
    },
    "langgraph": {
        "Retry": {"with_retry"},
        "Termination": set(),
        "Checkpoint": set(),
    },
    "crewai": {
        "Retry": set(),
        "Termination": set(),
        "Checkpoint": {"from_checkpoint", "checkpoint", "acheckpoint"},
    },
    "metagpt": {
        "Retry": set(),
        "Termination": {"_set_react_mode"},
        "Checkpoint": set(),
    },
}

KEYWORD_RULES = {
    "langchain": {
        "Retry": {"retry_policy"},
        "Termination": set(),
        "Checkpoint": {"checkpointer"},
    },
    "langgraph": {
        "Retry": {"retry_policy"},
        "Termination": set(),
        "Checkpoint": {"checkpointer"},
    },
    "llamaindex": {
        "Retry": {"retry_policy"},
        "Termination": {"max_iterations", "early_stopping_method"},
        "Checkpoint": set(),
    },
    "crewai": {
        "Retry": {"guardrail_max_retries"},
        "Termination": {"max_iter", "max_execution_time"},
        "Checkpoint": {"max_checkpoints"},
    },
    "autogen": {
        "Retry": set(),
        "Termination": {
            "termination_condition",
            "max_turns",
            "max_consecutive_auto_reply",
            "is_termination_msg",
        },
        "Checkpoint": set(),
    },
    "haystack": {
        "Retry": set(),
        "Termination": {"max_runs_per_component"},
        "Checkpoint": set(),
    },
    "semantic_kernel": {
        "Retry": set(),
        "Termination": {"maximum_iterations"},
        "Checkpoint": set(),
    },
    "metagpt": {
        "Retry": set(),
        "Termination": {"max_react_loop"},
        "Checkpoint": set(),
    },
}

DICT_KEY_RULES = {
    "langchain": {
        "Termination": {"recursion_limit"},
    },
    "langgraph": {
        "Termination": {"recursion_limit"},
    },
}

NAME_RULES = {
    "langchain": {
        "Termination": {"GraphRecursionError"},
    },
    "langgraph": {
        "Termination": {"GraphRecursionError"},
    },
}


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
    api: str
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


def dict_key_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def is_official_module(module: str | None, roots: tuple[str, ...]) -> bool:
    return bool(module) and module.startswith(roots)


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.source = source
        self.lines = source.splitlines()
        self.roots = OFFICIAL_ROOTS_BY_FRAMEWORK[framework]
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.official_import_seen = False
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, category: str, rule: str, api: str, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                api=api,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_official_module(alias.name, self.roots):
                self.official_import_seen = True
                self.official_module_aliases.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if is_official_module(module, self.roots):
            self.official_import_seen = True
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

    def add_call_rule(self, last: str, node: ast.AST) -> None:
        for category, names in CALL_RULES.get(self.framework, {}).items():
            if last in names:
                self.add(category, f"{last}_call", last, node)

    def add_method_rule(self, full: str | None, last: str, node: ast.AST) -> None:
        if not self.official_import_seen:
            return
        for category, names in METHOD_RULES.get(self.framework, {}).items():
            if last not in names:
                continue
            if last == "from_checkpoint" and full != "Crew.from_checkpoint":
                continue
            self.add(category, f"{last}_method", last, node)

    def add_keyword_rules(self, node: ast.Call) -> None:
        if not self.official_import_seen:
            return
        for kw in node.keywords:
            if kw.arg is None:
                continue
            for category, names in KEYWORD_RULES.get(self.framework, {}).items():
                if kw.arg in names:
                    self.add(category, f"{kw.arg}_keyword", kw.arg, node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        confirmed = self.is_confirmed_official_call(full, last)

        if confirmed and last:
            self.add_call_rule(last, node)

        if last:
            self.add_method_rule(full, last, node)

        self.add_keyword_rules(node)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if self.official_import_seen:
            for key in node.keys:
                name = dict_key_name(key) if key is not None else None
                for category, names in DICT_KEY_RULES.get(self.framework, {}).items():
                    if name in names:
                        self.add(category, f"{name}_dict_key", name or "", node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self.official_import_seen:
            for category, names in NAME_RULES.get(self.framework, {}).items():
                if node.id in names:
                    self.add(category, f"{node.id}_name", node.id, node)
        self.generic_visit(node)


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[Evidence], str | None]:
    source = read_text(path)
    if source is None:
        return [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], "syntax_error"
    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.evidence, None


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    evidence: list[Evidence] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            file_evidence, error = analyze_file(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        evidence.extend(file_evidence)
        if error == "read_error":
            read_errors += 1
        elif error == "syntax_error":
            syntax_errors += 1

    categories = {ev.category for ev in evidence}
    has_retry = "Retry" in categories
    has_termination = "Termination" in categories
    has_checkpoint = "Checkpoint" in categories
    defect = not (has_retry or has_termination or has_checkpoint)

    rules_by_category: dict[str, set[str]] = {"Retry": set(), "Termination": set(), "Checkpoint": set()}
    for ev in evidence:
        rules_by_category.setdefault(ev.category, set()).add(ev.rule)

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_retry": has_retry,
        "has_termination": has_termination,
        "has_checkpoint": has_checkpoint,
        "bp3_defect": defect,
        "retry_rules": ";".join(sorted(rules_by_category.get("Retry", set()))),
        "termination_rules": ";".join(sorted(rules_by_category.get("Termination", set()))),
        "checkpoint_rules": ";".join(sorted(rules_by_category.get("Checkpoint", set()))),
        "evidence": evidence,
    }


def list_repos(base: Path, frameworks: list[str], limit: int = 0) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        fw_entries = [entry for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name) if entry.is_dir()]
        if limit:
            fw_entries = fw_entries[:limit]
        for entry in fw_entries:
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
    repo_csv = out_dir / "bp3_repo_results.csv"
    evidence_csv = out_dir / "bp3_evidence.csv"
    summary_csv = out_dir / "bp3_summary_by_framework.csv"
    scaled_csv = out_dir / "bp3_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "has_retry",
                "has_termination",
                "has_checkpoint",
                "bp3_defect",
                "retry_rules",
                "termination_rules",
                "checkpoint_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "category", "rule", "api", "file", "line", "code"],
        )
        writer.writeheader()
        for row in results:
            seen: set[tuple[str, str, str, str, str, int]] = set()
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
                        "api": ev.api,
                        "file": ev.file,
                        "line": ev.line,
                        "code": ev.code,
                    }
                )

    summary_rows = []
    scaled_rows = []
    for fw in sorted({row["framework"] for row in results}):
        rows = [row for row in results if row["framework"] == fw]
        population = len(rows)
        retry = sum(1 for row in rows if row["has_retry"])
        termination = sum(1 for row in rows if row["has_termination"])
        checkpoint = sum(1 for row in rows if row["has_checkpoint"])
        any_mechanism = sum(1 for row in rows if row["has_retry"] or row["has_termination"] or row["has_checkpoint"])
        defect = sum(1 for row in rows if row["bp3_defect"])
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_retry": retry,
                "repos_with_termination": termination,
                "repos_with_checkpoint": checkpoint,
                "repos_with_any_self_recovery": any_mechanism,
                "bp3_defect_repos": defect,
                "py_files": py_files,
                "read_errors": read_errors,
                "syntax_errors": syntax_errors,
            }
        )

        target = target_sample_size(population)
        scaled_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "target_report_repos": target,
                "scaled_repos_with_retry": scale_count(retry, population, target),
                "scaled_repos_with_termination": scale_count(termination, population, target),
                "scaled_repos_with_checkpoint": scale_count(checkpoint, population, target),
                "scaled_repos_with_any_self_recovery": scale_count(any_mechanism, population, target),
                "scaled_bp3_defect_repos": scale_count(defect, population, target),
            }
        )

    for path, rows in [(summary_csv, summary_rows), (scaled_csv, scaled_rows)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            if not rows:
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp3_self_recovery")
    parser.add_argument("--framework", action="append")
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

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, limit=args.limit)

    print(f"Analyzing {len(repos)} repositories: {', '.join(frameworks)}", file=sys.stderr)
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
