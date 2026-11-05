#!/usr/bin/env python3
"""Detect BP5 overly verbose tool descriptions in downstream repositories.

The detector is restricted to framework-official tool APIs and static string
values that can be resolved from Python AST. A repository is counted once for
the defect if any tool description is longer than THRESHOLD characters.
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


THRESHOLD = 600

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
    "crewai": ("crewai", "crewai_tools"),
    "autogen": ("autogen", "autogen_agentchat", "autogen_core", "autogen_ext"),
    "haystack": ("haystack",),
    "semantic_kernel": ("semantic_kernel",),
    "camel": ("camel",),
    "metagpt": ("metagpt",),
    "autogpt": ("autogpt",),
}

# APIs where description= is an official tool-description field.
DESCRIPTION_CALL_APIS = {
    "langchain": {"Tool", "StructuredTool", "tool", "from_function"},
    "langgraph": {"Tool", "StructuredTool", "tool", "from_function"},
    "llamaindex": {"FunctionTool", "from_defaults", "ToolMetadata"},
    "crewai": {"tool", "BaseTool", "Tool"},
    "autogen": {"FunctionTool", "register_for_llm"},
    "haystack": {"Tool", "ComponentTool", "PipelineTool", "tool"},
    "semantic_kernel": {"kernel_function", "KernelFunctionFromPrompt", "add_function"},
    "camel": {"FunctionTool"},
    "metagpt": set(),
    "autogpt": set(),
}

# APIs that can derive a tool description from the bound function docstring.
DOCSTRING_TOOL_APIS = {
    "langchain": {"tool", "from_function"},
    "langgraph": {"tool", "from_function"},
    "llamaindex": {"from_defaults", "FunctionTool"},
    "crewai": {"tool"},
    "autogen": {"FunctionTool"},
    "haystack": {"tool"},
    "semantic_kernel": {"kernel_function"},
    "camel": {"FunctionTool"},
    "metagpt": set(),
    "autogpt": set(),
}

BASE_TOOL_CLASS_NAMES = {
    "langchain": {"BaseTool"},
    "langgraph": {"BaseTool"},
    "crewai": {"BaseTool"},
}


@dataclass(frozen=True)
class Evidence:
    framework: str
    repo: str
    category: str
    rule: str
    api: str
    description_length: int
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


def base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = attr_chain(node)
        return parts[-1] if parts else None
    return None


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
        self.string_vars: dict[str, str] = {}
        self.function_docstrings: dict[str, tuple[str, int]] = {}
        self.evidence: list[Evidence] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

    def add(self, rule: str, api: str, description: str, node: ast.AST) -> None:
        length = len(description)
        category = "OverlyVerboseToolDescription" if length > THRESHOLD else "ToolDescription"
        self.evidence.append(
            Evidence(
                framework=self.framework,
                repo=self.repo,
                category=category,
                rule=rule,
                api=api,
                description_length=length,
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

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if is_official_module(module, self.roots):
            self.official_import_seen = True
            for alias in node.names:
                local = alias.asname or alias.name
                self.alias_to_qualified[local] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def static_string(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    return None
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.static_string(node.left)
            right = self.static_string(node.right)
            if left is not None and right is not None:
                return left + right
        if isinstance(node, ast.Name):
            return self.string_vars.get(node.id)
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.static_string(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value is None:
                    self.string_vars.pop(target.id, None)
                else:
                    self.string_vars[target.id] = value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            value = self.static_string(node.value)
            if value is None:
                self.string_vars.pop(node.target.id, None)
            else:
                self.string_vars[node.target.id] = value
        self.generic_visit(node)

    def is_confirmed_official_call(self, full: str | None, last: str | None) -> bool:
        if not last:
            return False
        if last in self.alias_to_qualified:
            return True
        if full:
            parts = full.split(".")
            if not parts:
                return False
            if parts[0] in self.official_module_aliases:
                return True
            if parts[0] in self.alias_to_qualified:
                return True
        return False

    def decorator_api(self, deco: ast.AST) -> tuple[str | None, ast.Call | None, bool]:
        if isinstance(deco, ast.Call):
            full, last = call_name(deco.func)
            return last, deco, self.is_confirmed_official_call(full, last)
        full, last = call_name(deco)
        return last, None, self.is_confirmed_official_call(full, last)

    def function_name_arg(self, node: ast.Call) -> str | None:
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Name):
                return first.id
        for kw in node.keywords:
            if kw.arg in {"func", "fn", "function"} and isinstance(kw.value, ast.Name):
                return kw.value.id
        return None

    def add_description_kw(self, api: str, node: ast.Call) -> None:
        if api not in DESCRIPTION_CALL_APIS.get(self.framework, set()):
            return
        for kw in node.keywords:
            if kw.arg != "description":
                continue
            description = self.static_string(kw.value)
            if description is not None:
                self.add(f"{api}_description_keyword", api, description, node)

    def add_docstring_arg(self, api: str, node: ast.Call) -> None:
        if api not in DOCSTRING_TOOL_APIS.get(self.framework, set()):
            return
        fn_name = self.function_name_arg(node)
        if not fn_name or fn_name not in self.function_docstrings:
            return
        docstring, lineno = self.function_docstrings[fn_name]
        proxy = ast.copy_location(ast.Pass(), node)
        setattr(proxy, "lineno", lineno)
        self.add(f"{api}_function_docstring", api, docstring, proxy)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        docstring = ast.get_docstring(node, clean=False)
        if docstring is not None:
            self.function_docstrings[node.name] = (docstring, node.lineno)

        for deco in node.decorator_list:
            api, call, confirmed = self.decorator_api(deco)
            if not confirmed or not api:
                continue
            if call is not None:
                self.add_description_kw(api, call)
            if api in DOCSTRING_TOOL_APIS.get(self.framework, set()) and docstring is not None:
                self.add(f"{api}_decorated_function_docstring", api, docstring, node)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def class_inherits_base_tool(self, node: ast.ClassDef) -> bool:
        wanted = BASE_TOOL_CLASS_NAMES.get(self.framework, set())
        if not wanted:
            return False
        for base in node.bases:
            name = base_name(base)
            if name not in wanted:
                continue
            if name in self.alias_to_qualified:
                return True
            if self.official_import_seen:
                return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        if self.class_inherits_base_tool(node):
            for stmt in node.body:
                value_node: ast.AST | None = None
                if isinstance(stmt, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "description" for t in stmt.targets):
                        value_node = stmt.value
                elif isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id == "description":
                        value_node = stmt.value
                if value_node is None:
                    continue
                description = self.static_string(value_node)
                if description is not None:
                    self.add("BaseTool_class_description", "BaseTool", description, stmt)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last:
            self.add_description_kw(last, node)
            self.add_docstring_arg(last, node)
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

    max_description_length = max((ev.description_length for ev in evidence), default=0)
    defect = any(ev.category == "OverlyVerboseToolDescription" for ev in evidence)
    rules = sorted({ev.rule for ev in evidence if ev.category == "OverlyVerboseToolDescription"})

    return {
        "framework": framework,
        "repo": repo_name,
        "repo_path": repo_path_str,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "has_static_tool_description": bool(evidence),
        "max_description_length": max_description_length,
        "bp5_verbose_tool_description_defect": defect,
        "verbose_rules": ";".join(rules),
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
    repo_csv = out_dir / "bp5_verbose_tool_description_repo_results.csv"
    evidence_csv = out_dir / "bp5_verbose_tool_description_evidence.csv"
    summary_csv = out_dir / "bp5_verbose_tool_description_summary_by_framework.csv"
    scaled_csv = out_dir / "bp5_verbose_tool_description_scaled_summary_by_framework.csv"

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "read_errors",
                "syntax_errors",
                "has_static_tool_description",
                "max_description_length",
                "bp5_verbose_tool_description_defect",
                "verbose_rules",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    with evidence_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "category",
                "rule",
                "api",
                "description_length",
                "file",
                "line",
                "code",
            ],
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
                        "description_length": ev.description_length,
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
        static_desc = sum(1 for row in rows if row["has_static_tool_description"])
        defect = sum(1 for row in rows if row["bp5_verbose_tool_description_defect"])
        max_len = max((int(row["max_description_length"]) for row in rows), default=0)
        py_files = sum(int(row["py_files"]) for row in rows)
        read_errors = sum(int(row["read_errors"]) for row in rows)
        syntax_errors = sum(int(row["syntax_errors"]) for row in rows)

        summary_rows.append(
            {
                "framework": fw,
                "population_repos": population,
                "repos_with_static_tool_description": static_desc,
                "bp5_verbose_tool_description_defect_repos": defect,
                "max_detected_description_length": max_len,
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
                "scaled_repos_with_static_tool_description": scale_count(static_desc, population, target),
                "scaled_bp5_verbose_tool_description_defect_repos": scale_count(defect, population, target),
                "max_detected_description_length": max_len,
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
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp5_verbose_tool_description")
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
