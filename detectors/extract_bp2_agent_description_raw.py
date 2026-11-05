#!/usr/bin/env python3
"""Extract raw agent-description data for BP2 LLM-assisted analysis.

This script does not classify defects. It collects static fields from official
agent-construction APIs and prompt builders so an LLM/human validator can judge
description completeness and semantic consistency.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
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
    "crewai": ("crewai", "crewai_tools"),
    "autogen": ("autogen", "autogen_agentchat", "autogen_core", "autogen_ext"),
    "haystack": ("haystack",),
    "semantic_kernel": ("semantic_kernel",),
    "camel": ("camel",),
    "metagpt": ("metagpt",),
    "autogpt": ("autogpt",),
}

AGENT_APIS = {
    "langchain": {
        "create_agent",
        "create_react_agent",
        "initialize_agent",
        "create_structured_chat_agent",
        "create_tool_calling_agent",
        "create_openai_functions_agent",
        "AgentExecutor",
    },
    "langgraph": {"create_agent", "create_react_agent"},
    "llamaindex": {"FunctionAgent", "ReActAgent", "OpenAIAgent", "AgentRunner", "AgentWorkflow", "from_tools"},
    "crewai": {"Agent"},
    "autogen": {"AssistantAgent", "UserProxyAgent", "ConversableAgent"},
    "haystack": {"Agent"},
    "semantic_kernel": {"ChatCompletionAgent", "OpenAIAssistantAgent", "AssistantAgent"},
    "camel": {"ChatAgent", "RolePlaying"},
    "metagpt": {"Role"},
    "autogpt": set(),
}

FIELD_NAMES = {
    "name",
    "role",
    "goal",
    "backstory",
    "profile",
    "description",
    "desc",
    "constraints",
    "constraint",
    "context",
    "system_prompt",
    "system_message",
    "instructions",
    "instruction",
    "prompt",
    "state_modifier",
    "messages_modifier",
}

POSITIONAL_FIELDS = {
    "AssistantAgent": {0: "name"},
    "UserProxyAgent": {0: "name"},
    "ConversableAgent": {0: "name"},
    "ChatAgent": {0: "system_message"},
    "Role": {0: "profile"},
}

PROMPT_BUILDERS = {"from_messages", "from_template", "PromptTemplate", "ChatPromptTemplate", "SystemMessage"}


@dataclass(frozen=True)
class AgentRecord:
    framework: str
    repo: str
    api: str
    file: str
    line: int
    code: str
    fields: dict[str, str]
    field_exprs: dict[str, str]
    text_blob: str


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


def unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)[:500]
    except Exception:
        return ast.dump(node)[:500]


class FileAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.rel_path = path.relative_to(repo_dir).as_posix()
        self.lines = source.splitlines()
        self.roots = OFFICIAL_ROOTS_BY_FRAMEWORK[framework]
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.string_vars: dict[str, str] = {}
        self.prompt_vars: dict[str, str] = {}
        self.records: list[AgentRecord] = []

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:300]
        return ""

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
            return self.string_vars.get(node.id) or self.prompt_vars.get(node.id)
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_official_module(alias.name, self.roots):
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
            if parts and (parts[0] in self.official_module_aliases or parts[0] in self.alias_to_qualified):
                return True
        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.static_string(node.value)
        prompt = self.extract_prompt_text(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value is not None:
                    self.string_vars[target.id] = value
                if prompt is not None:
                    self.prompt_vars[target.id] = prompt
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            value = self.static_string(node.value)
            prompt = self.extract_prompt_text(node.value)
            if value is not None:
                self.string_vars[node.target.id] = value
            if prompt is not None:
                self.prompt_vars[node.target.id] = prompt
        self.generic_visit(node)

    def extract_prompt_text(self, node: ast.AST | None) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        full, last = call_name(node.func)
        if last not in PROMPT_BUILDERS:
            return None
        chunks: list[str] = []
        for arg in node.args:
            self.collect_strings(arg, chunks)
        for kw in node.keywords:
            self.collect_strings(kw.value, chunks)
        return "\n".join(chunks) if chunks else None

    def collect_strings(self, node: ast.AST, chunks: list[str]) -> None:
        value = self.static_string(node)
        if value is not None:
            chunks.append(value)
            return
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self.collect_strings(elt, chunks)
        elif isinstance(node, ast.Dict):
            for value_node in node.values:
                self.collect_strings(value_node, chunks)
        elif isinstance(node, ast.Call):
            prompt = self.extract_prompt_text(node)
            if prompt:
                chunks.append(prompt)

    def field_value(self, node: ast.AST | None) -> tuple[str | None, str]:
        value = self.static_string(node)
        return value, unparse(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last in AGENT_APIS.get(self.framework, set()):
            self.add_agent_record(last or "", node)
        self.generic_visit(node)

    def add_agent_record(self, api: str, node: ast.Call) -> None:
        fields: dict[str, str] = {}
        exprs: dict[str, str] = {}

        for pos, field in POSITIONAL_FIELDS.get(api, {}).items():
            if pos < len(node.args):
                value, expr = self.field_value(node.args[pos])
                if value is not None:
                    fields[field] = value
                exprs[field] = expr

        for kw in node.keywords:
            if kw.arg not in FIELD_NAMES:
                continue
            value, expr = self.field_value(kw.value)
            if value is not None:
                fields[kw.arg] = value
            exprs[kw.arg] = expr

        if not fields and not exprs:
            return

        text_blob = "\n".join(f"{k}: {v}" for k, v in sorted(fields.items()))
        self.records.append(
            AgentRecord(
                framework=self.framework,
                repo=self.repo,
                api=api,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
                fields=fields,
                field_exprs=exprs,
                text_blob=text_blob,
            )
        )


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[AgentRecord], str | None]:
    source = read_text(path)
    if source is None:
        return [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], "syntax_error"
    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.records, None


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    records: list[AgentRecord] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            file_records, error = analyze_file(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        records.extend(file_records)
        if error == "read_error":
            read_errors += 1
        elif error == "syntax_error":
            syntax_errors += 1
    return {
        "framework": framework,
        "repo": repo_name,
        "py_files": py_count,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "agent_records": records,
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


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "bp2_agent_description_raw.jsonl"
    repo_csv = out_dir / "bp2_agent_description_repo_summary.csv"
    fw_csv = out_dir / "bp2_agent_description_summary_by_framework.csv"

    with raw_path.open("w", encoding="utf-8") as f:
        for row in results:
            for record in row["agent_records"]:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "py_files", "read_errors", "syntax_errors", "agent_record_count"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "framework": row["framework"],
                    "repo": row["repo"],
                    "py_files": row["py_files"],
                    "read_errors": row["read_errors"],
                    "syntax_errors": row["syntax_errors"],
                    "agent_record_count": len(row["agent_records"]),
                }
            )

    summary_rows = []
    for fw in sorted({row["framework"] for row in results}):
        rows = [row for row in results if row["framework"] == fw]
        repos_with_records = sum(1 for row in rows if row["agent_records"])
        record_count = sum(len(row["agent_records"]) for row in rows)
        summary_rows.append(
            {
                "framework": fw,
                "population_repos": len(rows),
                "repos_with_agent_description_raw": repos_with_records,
                "agent_record_count": record_count,
                "py_files": sum(int(row["py_files"]) for row in rows),
                "read_errors": sum(int(row["read_errors"]) for row in rows),
                "syntax_errors": sum(int(row["syntax_errors"]) for row in rows),
            }
        )
    with fw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp2_agent_description_raw")
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
