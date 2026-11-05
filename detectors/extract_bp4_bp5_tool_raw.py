#!/usr/bin/env python3
"""Extract raw tool data for BP4/BP5 LLM-assisted analyses.

Outputs:
- bp4_tool_group_raw.jsonl: tool collections bound to one agent/tool node.
- bp5_tool_io_description_raw.jsonl: tool definitions with descriptions,
  signatures, and schema expressions.

The script does not classify functional overlap or missing I/O specification.
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

TOOL_APIS = {
    "langchain": {"tool", "Tool", "StructuredTool", "from_function"},
    "langgraph": {"tool", "Tool", "StructuredTool", "from_function", "ToolNode"},
    "llamaindex": {"FunctionTool", "from_defaults", "QueryEngineTool", "ToolMetadata"},
    "crewai": {"tool", "BaseTool", "Tool"},
    "autogen": {"FunctionTool", "register_for_llm"},
    "haystack": {"Tool", "ComponentTool", "PipelineTool", "tool"},
    "semantic_kernel": {"kernel_function", "KernelFunctionFromPrompt", "add_function"},
    "camel": {"FunctionTool"},
    "metagpt": set(),
    "autogpt": set(),
}

AGENT_TOOL_SPECS = {
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
    "crewai": {"Agent": ({"tools"}, set())},
    "autogen": {
        "AssistantAgent": ({"tools", "function_map"}, set()),
        "ConversableAgent": ({"tools", "function_map"}, set()),
        "UserProxyAgent": ({"function_map"}, set()),
    },
    "haystack": {"Agent": ({"tools"}, set()), "ToolInvoker": ({"tools"}, set())},
    "semantic_kernel": {"ChatCompletionAgent": ({"plugins"}, set()), "OpenAIAssistantAgent": ({"plugins"}, set())},
    "camel": {"ChatAgent": ({"tools"}, set())},
    "metagpt": {"DataInterpreter": ({"tools"}, set())},
    "autogpt": {},
}


@dataclass(frozen=True)
class ToolRecord:
    framework: str
    repo: str
    api: str
    file: str
    line: int
    code: str
    name: str
    description: str
    signature: str
    input_schema: str
    output_schema: str
    source_expr: str


@dataclass(frozen=True)
class ToolGroupRecord:
    framework: str
    repo: str
    api: str
    file: str
    line: int
    code: str
    tool_count: int
    tools: list[dict[str, str]]


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
        return ast.unparse(node)[:1000]
    except Exception:
        return ast.dump(node)[:1000]


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
        self.function_docs: dict[str, str] = {}
        self.function_sigs: dict[str, str] = {}
        self.function_returns: dict[str, str] = {}
        self.tool_vars: dict[str, ToolRecord] = {}
        self.tool_list_vars: dict[str, list[ast.AST]] = {}
        self.tool_records: list[ToolRecord] = []
        self.group_records: list[ToolGroupRecord] = []

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
            return self.string_vars.get(node.id)
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if is_official_module(alias.name, self.roots):
                self.official_module_aliases.add(alias.asname or alias.name.split(".")[0])
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

    def function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args: list[str] = []
        all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in all_args:
            ann = f": {unparse(arg.annotation)}" if arg.annotation else ""
            args.append(f"{arg.arg}{ann}")
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        ret = f" -> {unparse(node.returns)}" if node.returns else ""
        return f"{node.name}({', '.join(args)}){ret}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.function_docs[node.name] = ast.get_docstring(node, clean=False) or ""
        self.function_sigs[node.name] = self.function_signature(node)
        self.function_returns[node.name] = unparse(node.returns)

        for deco in node.decorator_list:
            api, call, confirmed = self.decorator_api(deco)
            if confirmed and api in TOOL_APIS.get(self.framework, set()):
                self.add_decorated_tool_record(api, node, call)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def decorator_api(self, deco: ast.AST) -> tuple[str | None, ast.Call | None, bool]:
        if isinstance(deco, ast.Call):
            full, last = call_name(deco.func)
            return last, deco, self.is_confirmed_official_call(full, last)
        full, last = call_name(deco)
        return last, None, self.is_confirmed_official_call(full, last)

    def add_decorated_tool_record(self, api: str, fn: ast.FunctionDef | ast.AsyncFunctionDef, deco_call: ast.Call | None) -> None:
        name = fn.name
        description = self.function_docs.get(fn.name, "")
        if deco_call is not None:
            if deco_call.args:
                first = self.static_string(deco_call.args[0])
                if first:
                    name = first
            for kw in deco_call.keywords:
                if kw.arg == "name":
                    name = self.static_string(kw.value) or name
                if kw.arg == "description":
                    description = self.static_string(kw.value) or description
        record = ToolRecord(
            framework=self.framework,
            repo=self.repo,
            api=api,
            file=self.rel_path,
            line=fn.lineno,
            code=self.line_text(fn.lineno),
            name=name,
            description=description,
            signature=self.function_sigs.get(fn.name, ""),
            input_schema="",
            output_schema=self.function_returns.get(fn.name, ""),
            source_expr=fn.name,
        )
        self.tool_records.append(record)
        self.tool_vars[fn.name] = record

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        base_names = {call_name(base)[1] for base in node.bases if isinstance(base, (ast.Name, ast.Attribute))}
        if "BaseTool" in base_names:
            name = node.name
            description = ""
            input_schema = ""
            signature = ""
            output_schema = ""
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name in {"_run", "_arun", "run"}:
                    signature = self.function_signature(stmt)
                    output_schema = unparse(stmt.returns)
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            if target.id == "name":
                                name = self.static_string(stmt.value) or name
                            elif target.id == "description":
                                description = self.static_string(stmt.value) or description
                            elif target.id in {"args_schema", "input_schema"}:
                                input_schema = unparse(stmt.value)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == "name":
                        name = self.static_string(stmt.value) or name
                    elif stmt.target.id == "description":
                        description = self.static_string(stmt.value) or description
                    elif stmt.target.id in {"args_schema", "input_schema"}:
                        input_schema = unparse(stmt.value)
            record = ToolRecord(
                framework=self.framework,
                repo=self.repo,
                api="BaseTool",
                file=self.rel_path,
                line=node.lineno,
                code=self.line_text(node.lineno),
                name=name,
                description=description,
                signature=signature,
                input_schema=input_schema,
                output_schema=output_schema,
                source_expr=node.name,
            )
            self.tool_records.append(record)
            self.tool_vars[node.name] = record
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.static_string(node.value)
        tool_record = self.extract_tool_record_from_call(node.value)
        list_elts = self.collection_elts(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if value is not None:
                    self.string_vars[target.id] = value
                if tool_record is not None:
                    self.tool_vars[target.id] = tool_record
                if list_elts is not None:
                    self.tool_list_vars[target.id] = list_elts
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            value = self.static_string(node.value)
            tool_record = self.extract_tool_record_from_call(node.value)
            list_elts = self.collection_elts(node.value)
            if value is not None:
                self.string_vars[node.target.id] = value
            if tool_record is not None:
                self.tool_vars[node.target.id] = tool_record
            if list_elts is not None:
                self.tool_list_vars[node.target.id] = list_elts
        self.generic_visit(node)

    def collection_elts(self, node: ast.AST | None) -> list[ast.AST] | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return list(node.elts)
        if isinstance(node, ast.Name):
            return self.tool_list_vars.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.collection_elts(node.left)
            right = self.collection_elts(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def visit_Call(self, node: ast.Call) -> None:
        tool_record = self.extract_tool_record_from_call(node)
        if tool_record is not None:
            self.tool_records.append(tool_record)

        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last in AGENT_TOOL_SPECS.get(self.framework, {}):
            self.add_tool_group_record(last or "", node)

        self.generic_visit(node)

    def extract_tool_record_from_call(self, node: ast.AST | None) -> ToolRecord | None:
        if not isinstance(node, ast.Call):
            return None
        full, last = call_name(node.func)
        if not (self.is_confirmed_official_call(full, last) and last in TOOL_APIS.get(self.framework, set())):
            return None

        name = ""
        description = ""
        input_schema = ""
        output_schema = ""
        signature = ""
        source_expr = unparse(node)

        fn_name = self.function_arg_name(node)
        if fn_name:
            name = fn_name
            description = self.function_docs.get(fn_name, "")
            signature = self.function_sigs.get(fn_name, "")
            output_schema = self.function_returns.get(fn_name, "")

        if last in {"Tool", "StructuredTool"} and node.args:
            first = self.static_string(node.args[0])
            if first:
                name = first

        for kw in node.keywords:
            if kw.arg in {"name", "tool_name"}:
                name = self.static_string(kw.value) or name
            elif kw.arg in {"description", "desc"}:
                description = self.static_string(kw.value) or description
            elif kw.arg in {"args_schema", "input_schema", "parameters", "args", "schema"}:
                input_schema = unparse(kw.value)
            elif kw.arg in {"return_schema", "output_schema", "response_schema"}:
                output_schema = unparse(kw.value)
            elif kw.arg in {"func", "fn", "function"} and isinstance(kw.value, ast.Name):
                fn = kw.value.id
                name = name or fn
                description = description or self.function_docs.get(fn, "")
                signature = signature or self.function_sigs.get(fn, "")
                output_schema = output_schema or self.function_returns.get(fn, "")

        if not (name or description or signature or input_schema or output_schema):
            return None
        return ToolRecord(
            framework=self.framework,
            repo=self.repo,
            api=last or "",
            file=self.rel_path,
            line=getattr(node, "lineno", 0),
            code=self.line_text(getattr(node, "lineno", 0)),
            name=name,
            description=description,
            signature=signature,
            input_schema=input_schema,
            output_schema=output_schema,
            source_expr=source_expr,
        )

    def function_arg_name(self, node: ast.Call) -> str:
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
        return ""

    def add_tool_group_record(self, api: str, node: ast.Call) -> None:
        keywords, positions = AGENT_TOOL_SPECS[self.framework][api]
        candidates: list[ast.AST] = []
        for pos in positions:
            if pos < len(node.args):
                candidates.append(node.args[pos])
        for kw in node.keywords:
            if kw.arg in keywords:
                candidates.append(kw.value)

        tools: list[dict[str, str]] = []
        for candidate in candidates:
            for elt in self.collection_elts(candidate) or []:
                tools.append(self.tool_info_from_expr(elt))
        if len(tools) < 2:
            return
        self.group_records.append(
            ToolGroupRecord(
                framework=self.framework,
                repo=self.repo,
                api=api,
                file=self.rel_path,
                line=getattr(node, "lineno", 0),
                code=self.line_text(getattr(node, "lineno", 0)),
                tool_count=len(tools),
                tools=tools,
            )
        )

    def tool_info_from_expr(self, node: ast.AST) -> dict[str, str]:
        expr = unparse(node)
        record = None
        if isinstance(node, ast.Name):
            record = self.tool_vars.get(node.id)
            if record is None and node.id in self.function_docs:
                return {
                    "expr": expr,
                    "name": node.id,
                    "description": self.function_docs.get(node.id, ""),
                    "signature": self.function_sigs.get(node.id, ""),
                    "input_schema": "",
                    "output_schema": self.function_returns.get(node.id, ""),
                }
        elif isinstance(node, ast.Call):
            record = self.extract_tool_record_from_call(node)
        if record is None:
            return {"expr": expr, "name": "", "description": "", "signature": "", "input_schema": "", "output_schema": ""}
        return {
            "expr": expr,
            "name": record.name,
            "description": record.description,
            "signature": record.signature,
            "input_schema": record.input_schema,
            "output_schema": record.output_schema,
        }


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[ToolRecord], list[ToolGroupRecord], str | None]:
    source = read_text(path)
    if source is None:
        return [], [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], [], "syntax_error"
    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.tool_records, analyzer.group_records, None


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    tool_records: list[ToolRecord] = []
    group_records: list[ToolGroupRecord] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            file_tools, file_groups, error = analyze_file(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        tool_records.extend(file_tools)
        group_records.extend(file_groups)
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
        "tool_records": tool_records,
        "tool_group_records": group_records,
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
    tool_path = out_dir / "bp5_tool_io_description_raw.jsonl"
    group_path = out_dir / "bp4_tool_group_raw.jsonl"
    repo_csv = out_dir / "bp4_bp5_tool_raw_repo_summary.csv"
    fw_csv = out_dir / "bp4_bp5_tool_raw_summary_by_framework.csv"

    with tool_path.open("w", encoding="utf-8") as f:
        for row in results:
            seen: set[tuple[str, str, str, str, int, str]] = set()
            for record in row["tool_records"]:
                key = (record.framework, record.repo, record.file, record.api, record.line, record.name)
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with group_path.open("w", encoding="utf-8") as f:
        for row in results:
            for record in row["tool_group_records"]:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["framework", "repo", "py_files", "read_errors", "syntax_errors", "tool_record_count", "tool_group_count"],
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
                    "tool_record_count": len(row["tool_records"]),
                    "tool_group_count": len(row["tool_group_records"]),
                }
            )

    summary_rows = []
    for fw in sorted({row["framework"] for row in results}):
        rows = [row for row in results if row["framework"] == fw]
        summary_rows.append(
            {
                "framework": fw,
                "population_repos": len(rows),
                "repos_with_tool_records": sum(1 for row in rows if row["tool_records"]),
                "repos_with_tool_groups": sum(1 for row in rows if row["tool_group_records"]),
                "tool_record_count": sum(len(row["tool_records"]) for row in rows),
                "tool_group_count": sum(len(row["tool_group_records"]) for row in rows),
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
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp4_bp5_tool_raw")
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
