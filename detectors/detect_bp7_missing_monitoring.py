#!/usr/bin/env python3
"""Detect BP7 missing monitoring mechanisms in downstream repositories.

Evidence categories include logging, tracing, callback hooks, and monitoring
integrations. A repository is counted as BP7 defect when no such evidence is
found in implementation Python files.
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

FRAMEWORK_ROOTS = (
    "langchain",
    "langchain_core",
    "langchain_classic",
    "langgraph",
    "llama_index",
    "llama_agents",
    "crewai",
    "crewai_tools",
    "autogen",
    "autogen_agentchat",
    "autogen_core",
    "haystack",
    "semantic_kernel",
    "camel",
    "metagpt",
    "autogpt",
)

MONITORING_ROOTS = {
    "logging",
    "loguru",
    "structlog",
    "sentry_sdk",
    "opentelemetry",
    "langsmith",
    "langfuse",
    "mlflow",
    "wandb",
    "phoenix",
    "openinference",
    "arize",
    "braintrust",
    "helicone",
}

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}

CALL_CATEGORY_RULES = {
    "Logging": {
        "basicConfig",
        "getLogger",
        "get_logger",
        "configure",
    },
    "Tracing": {
        "traceable",
        "tracing_context",
        "get_tracer",
        "TracerProvider",
        "LangChainTracer",
        "ConsoleCallbackHandler",
        "FileCallbackHandler",
        "StdOutCallbackHandler",
        "StreamingStdOutCallbackHandler",
        "LlamaDebugHandler",
        "TokenCountingHandler",
        "OpenInferenceCallbackHandler",
        "WandbCallbackHandler",
        "PhoenixCallbackHandler",
        "LoggingTracer",
        "OpenTelemetryTracer",
    },
    "Monitoring": {
        "init",
        "login",
        "watch",
        "finish",
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_metric",
        "log_metrics",
        "log_param",
        "log_params",
        "log_artifact",
        "log",
        "observe",
        "flush",
    },
    "Callback": {
        "CallbackManager",
        "BaseCallbackHandler",
        "CallbackHandler",
        "BaseTracer",
    },
}

CALLBACK_KEYWORDS = {"callbacks", "callback_manager", "callback", "step_callback", "task_callback"}
TRACING_ENV_KEYS = {
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "WANDB_PROJECT",
    "WANDB_API_KEY",
    "MLFLOW_TRACKING_URI",
    "PHOENIX_COLLECTOR_ENDPOINT",
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


def static_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def module_root(module: str) -> str:
    return module.split(".")[0]


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.lines = source.splitlines()
        self.alias_to_module: dict[str, str] = {}
        self.imported_names: dict[str, str] = {}
        self.logger_names: set[str] = set()
        self.framework_import_seen = False
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
            local = alias.asname or alias.name.split(".")[0]
            self.alias_to_module[local] = alias.name
            if alias.name.startswith(FRAMEWORK_ROOTS):
                self.framework_import_seen = True
            if module_root(alias.name) == "loguru" and local == "logger":
                self.logger_names.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module.startswith(FRAMEWORK_ROOTS):
            self.framework_import_seen = True
        for alias in node.names:
            local = alias.asname or alias.name
            self.imported_names[local] = f"{module}.{alias.name}"
            if module_root(module) == "loguru" and alias.name == "logger":
                self.logger_names.add(local)
            if alias.name in {"logger", "get_logger"} and module_root(module) in {"logging", "structlog"}:
                self.logger_names.add(local)
        self.generic_visit(node)

    def call_module_root(self, full: str | None, last: str | None) -> str | None:
        if not full:
            if last and last in self.imported_names:
                return module_root(self.imported_names[last])
            return None
        first = full.split(".")[0]
        if first in self.alias_to_module:
            return module_root(self.alias_to_module[first])
        if first in self.imported_names:
            return module_root(self.imported_names[first])
        return None

    def is_official_monitoring_call(self, full: str | None, last: str | None) -> bool:
        root = self.call_module_root(full, last)
        return bool(root and root in MONITORING_ROOTS)

    def visit_Assign(self, node: ast.Assign) -> None:
        full, last = call_name(node.value) if isinstance(node.value, ast.Call) else (None, None)
        if self.is_official_monitoring_call(full, last) and last in {"getLogger", "get_logger"}:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.logger_names.add(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        root = self.call_module_root(full, last)

        if root == "logging" and last in ({"basicConfig", "getLogger"} | LOG_METHODS):
            self.add("Logging", f"logging_{last}", node)

        if full:
            parts = full.split(".")
            if len(parts) >= 2 and parts[0] in self.logger_names and parts[-1] in LOG_METHODS:
                self.add("Logging", f"logger_{parts[-1]}", node)

        if self.is_official_monitoring_call(full, last):
            for category, names in CALL_CATEGORY_RULES.items():
                if last in names:
                    self.add(category, f"{root}_{last}", node)

        if last and last in self.imported_names:
            qualified = self.imported_names[last]
            root = module_root(qualified)
            if root in MONITORING_ROOTS and (
                last in CALL_CATEGORY_RULES["Tracing"]
                or last in CALL_CATEGORY_RULES["Monitoring"]
                or last in CALL_CATEGORY_RULES["Callback"]
            ):
                self.add("Tracing" if root not in {"sentry_sdk", "mlflow", "wandb"} else "Monitoring", f"{root}_{last}", node)
            if last in CALL_CATEGORY_RULES["Callback"]:
                self.add("Callback", f"{last}_call", node)
            if last in CALL_CATEGORY_RULES["Tracing"]:
                self.add("Tracing", f"{last}_call", node)

        if self.framework_import_seen:
            for kw in node.keywords:
                if kw.arg in CALLBACK_KEYWORDS:
                    self.add("Callback", f"{kw.arg}_keyword", node)

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            key_name = static_string(key)
            if key_name and self.is_tracing_env_key(key_name):
                self.add("Tracing", f"env_{key_name}", node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key_name = static_string(node.slice)
        if key_name and self.is_tracing_env_key(key_name):
            self.add("Tracing", f"env_{key_name}", node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        for base in node.bases:
            name = call_name(base)[1] if isinstance(base, (ast.Name, ast.Attribute)) else None
            if name in {"BaseCallbackHandler", "CallbackHandler", "BaseTracer"}:
                self.add("Callback", f"{name}_subclass", node)
        self.generic_visit(node)

    def is_tracing_env_key(self, key: str) -> bool:
        if key in TRACING_ENV_KEYS:
            return True
        return key.startswith(("OTEL_", "LANGFUSE_", "WANDB_", "MLFLOW_", "PHOENIX_"))


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
    has_logging = "Logging" in categories
    has_tracing = "Tracing" in categories
    has_callback = "Callback" in categories
    has_monitoring = "Monitoring" in categories
    any_mechanism = has_logging or has_tracing or has_callback or has_monitoring

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_logging": has_logging,
        "has_tracing": has_tracing,
        "has_callback": has_callback,
        "has_monitoring_integration": has_monitoring,
        "has_any_monitoring_mechanism": any_mechanism,
        "bp7_missing_monitoring_defect": not any_mechanism,
        "monitoring_rules": ";".join(sorted({ev.rule for ev in evidence})),
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
    repo_csv = out_dir / "bp7_missing_monitoring_repo_results.csv"
    evidence_csv = out_dir / "bp7_missing_monitoring_evidence.csv"
    summary_csv = out_dir / "bp7_missing_monitoring_summary_by_framework.csv"
    scaled_csv = out_dir / "bp7_missing_monitoring_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "has_logging",
                "has_tracing",
                "has_callback",
                "has_monitoring_integration",
                "has_any_monitoring_mechanism",
                "bp7_missing_monitoring_defect",
                "monitoring_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["framework", "repo", "category", "rule", "file", "line", "code"])
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
        logging = sum(1 for row in rows if row["has_logging"])
        tracing = sum(1 for row in rows if row["has_tracing"])
        callback = sum(1 for row in rows if row["has_callback"])
        monitoring = sum(1 for row in rows if row["has_monitoring_integration"])
        any_mech = sum(1 for row in rows if row["has_any_monitoring_mechanism"])
        defect = sum(1 for row in rows if row["bp7_missing_monitoring_defect"])
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_logging": logging,
                "repos_with_tracing": tracing,
                "repos_with_callback": callback,
                "repos_with_monitoring_integration": monitoring,
                "repos_with_any_monitoring_mechanism": any_mech,
                "bp7_missing_monitoring_defect_repos": defect,
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
                "scaled_repos_with_logging": scale_count(logging, population, target),
                "scaled_repos_with_tracing": scale_count(tracing, population, target),
                "scaled_repos_with_callback": scale_count(callback, population, target),
                "scaled_repos_with_monitoring_integration": scale_count(monitoring, population, target),
                "scaled_repos_with_any_monitoring_mechanism": scale_count(any_mech, population, target),
                "scaled_bp7_missing_monitoring_defect_repos": scale_count(defect, population, target),
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
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp7_missing_monitoring")
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
