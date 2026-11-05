#!/usr/bin/env python3
"""Detect BP4 excessive-tools evidence in downstream agent repositories.

The detector counts statically visible tool collections passed to official
agent/tool-binding APIs. It only reports repository-level BP4 defects when a
single detected binding exposes at least THRESHOLD tools.
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


THRESHOLD = 20

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

# Each spec is: API name -> keyword names and positional argument indexes that
# represent a tool/function/plugin collection for that API.
CALL_SPECS = {
    "langchain": {
        "create_agent": ({"tools"}, {1}),
        "initialize_agent": ({"tools"}, {0}),
        "create_react_agent": ({"tools"}, {1}),
        "create_structured_chat_agent": ({"tools"}, {1}),
        "create_tool_calling_agent": ({"tools"}, {1}),
        "create_openai_functions_agent": ({"tools"}, {1}),
        "AgentExecutor": ({"tools"}, set()),
    },
    "langgraph": {
        "create_agent": ({"tools"}, {1}),
        "create_react_agent": ({"tools"}, {1}),
        "ToolNode": ({"tools"}, {0}),
    },
    "llamaindex": {
        "FunctionAgent": ({"tools"}, set()),
        "ReActAgent": ({"tools"}, set()),
        "OpenAIAgent": ({"tools"}, set()),
        "AgentWorkflow": ({"tools", "agents"}, set()),
        "AgentRunner": ({"tools"}, set()),
        "from_tools": ({"tools"}, {0}),
        "from_tools_or_functions": ({"tools_or_functions"}, {0}),
    },
    "crewai": {
        "Agent": ({"tools"}, set()),
    },
    "autogen": {
        "AssistantAgent": ({"tools", "function_map"}, set()),
        "ConversableAgent": ({"tools", "function_map"}, set()),
        "UserProxyAgent": ({"function_map"}, set()),
    },
    "haystack": {
        "Agent": ({"tools"}, set()),
        "ToolInvoker": ({"tools"}, set()),
    },
    "semantic_kernel": {
        "ChatCompletionAgent": ({"plugins"}, set()),
        "OpenAIAssistantAgent": ({"plugins"}, set()),
    },
    "camel": {
        "ChatAgent": ({"tools"}, set()),
    },
    "metagpt": {
        "DataInterpreter": ({"tools"}, set()),
    },
    "autogpt": {},
}

METHOD_SPECS = {
    "camel": {
        "add_tools": ({"tools"}, {0}),
    },
    "llamaindex": {
        "from_tools": ({"tools"}, {0}),
        "from_tools_or_functions": ({"tools_or_functions"}, {0}),
    },
}


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
    api: str
    tool_count: int
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
        self.var_counts: dict[str, int] = {}
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, category: str, rule: str, api: str, tool_count: int, node: ast.AST) -> None:
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                api=api,
                tool_count=tool_count,
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

    def visit_Assign(self, node: ast.Assign) -> None:
        count = self.collection_count(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if count is None:
                    self.var_counts.pop(target.id, None)
                else:
                    self.var_counts[target.id] = count
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            count = self.collection_count(node.value) if node.value is not None else None
            if count is None:
                self.var_counts.pop(node.target.id, None)
            else:
                self.var_counts[node.target.id] = count
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            rhs = self.collection_count(node.value)
            if rhs is not None and node.target.id in self.var_counts:
                self.var_counts[node.target.id] += rhs
            else:
                self.var_counts.pop(node.target.id, None)
        self.generic_visit(node)

    def collection_count(self, node: ast.AST | None) -> int | None:
        if node is None:
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            total = 0
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    sub = self.collection_count(elt.value)
                    if sub is None:
                        return None
                    total += sub
                else:
                    total += 1
            return total
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                return None
            return len(node.keys)
        if isinstance(node, ast.Name):
            return self.var_counts.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.collection_count(node.left)
            right = self.collection_count(node.right)
            if left is not None and right is not None:
                return left + right
        return None

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

    def candidate_counts(self, node: ast.Call, keywords: set[str], positions: set[int]) -> list[tuple[str, int]]:
        counts: list[tuple[str, int]] = []
        for pos in positions:
            if pos < len(node.args):
                count = self.collection_count(node.args[pos])
                if count is not None:
                    counts.append((f"arg{pos}", count))
        for kw in node.keywords:
            if kw.arg in keywords:
                count = self.collection_count(kw.value)
                if count is not None:
                    counts.append((kw.arg, count))
        return counts

    def add_from_spec(self, node: ast.Call, api: str, specs: dict[str, tuple[set[str], set[int]]]) -> None:
        if api not in specs:
            return
        keywords, positions = specs[api]
        for source, count in self.candidate_counts(node, keywords, positions):
            category = "ExcessiveTools" if count >= THRESHOLD else "ToolList"
            rule = f"{api}_{source}_count_{count}"
            self.add(category, rule, api, count, node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        confirmed = self.is_confirmed_official_call(full, last)
        if confirmed and last:
            self.add_from_spec(node, last, CALL_SPECS.get(self.framework, {}))

        if self.official_import_seen and last:
            self.add_from_spec(node, last, METHOD_SPECS.get(self.framework, {}))

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

    max_tool_count = max((ev.tool_count for ev in evidence), default=0)
    excessive = any(ev.category == "ExcessiveTools" for ev in evidence)
    rules = sorted({ev.rule for ev in evidence if ev.category == "ExcessiveTools"})

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_static_tool_list": bool(evidence),
        "max_tool_count": max_tool_count,
        "bp4_excessive_tools_defect": excessive,
        "excessive_rules": ";".join(rules),
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
    repo_csv = out_dir / "bp4_excessive_tools_repo_results.csv"
    evidence_csv = out_dir / "bp4_excessive_tools_evidence.csv"
    summary_csv = out_dir / "bp4_excessive_tools_summary_by_framework.csv"
    scaled_csv = out_dir / "bp4_excessive_tools_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "has_static_tool_list",
                "max_tool_count",
                "bp4_excessive_tools_defect",
                "excessive_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "category", "rule", "api", "tool_count", "file", "line", "code"],
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
                        "tool_count": ev.tool_count,
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
        static_lists = sum(1 for row in rows if row["has_static_tool_list"])
        excessive = sum(1 for row in rows if row["bp4_excessive_tools_defect"])
        max_count = max((int(row["max_tool_count"]) for row in rows), default=0)
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_static_tool_list": static_lists,
                "bp4_excessive_tools_defect_repos": excessive,
                "max_detected_tool_count": max_count,
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
                "scaled_repos_with_static_tool_list": scale_count(static_lists, population, target),
                "scaled_bp4_excessive_tools_defect_repos": scale_count(excessive, population, target),
                "max_detected_tool_count": max_count,
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
    global THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp4_excessive_tools")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument("--threshold", type=int, default=THRESHOLD)
    parser.add_argument(
        "--exclude-non-impl-paths",
        action="store_true",
        help="Exclude test/example/documentation/demo/benchmark paths from analysis",
    )
    args = parser.parse_args()

    THRESHOLD = args.threshold

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, limit=args.limit)

    print(
        f"Analyzing {len(repos)} repositories: {', '.join(frameworks)}; threshold={THRESHOLD}",
        file=sys.stderr,
    )
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
