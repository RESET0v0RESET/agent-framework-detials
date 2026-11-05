#!/usr/bin/env python3
"""Extract framework-grounded tool-description evidence for RQ3.

This is a raw dataset extractor, not a defect classifier. It records custom
tool definitions, plain functions bound as tools, and official/prebuilt tool
references that are explicitly attached to an agent/tool executor.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rq3_filters import is_mock_like_name, iter_py_files


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
    "langchain": (
        "langchain",
        "langchain_core",
        "langchain_classic",
        "langchain_community",
        "langgraph",
    ),
    "langgraph": (
        "langgraph",
        "langchain",
        "langchain_core",
        "langchain_classic",
        "langchain_community",
    ),
    "llamaindex": ("llama_index", "llama_agents"),
    "crewai": ("crewai", "crewai_tools"),
    "autogen": (
        "autogen",
        "autogen_agentchat",
        "autogen_core",
        "autogen_ext",
    ),
    "haystack": ("haystack",),
    "semantic_kernel": ("semantic_kernel",),
    "camel": ("camel",),
    "metagpt": ("metagpt",),
    "autogpt": ("autogpt", "forge"),
}


TOOL_DEFINITION_APIS = {
    "langchain": {"tool", "Tool", "StructuredTool", "from_function"},
    "langgraph": {"tool", "Tool", "StructuredTool", "from_function"},
    "llamaindex": {
        "FunctionTool",
        "from_defaults",
        "QueryEngineTool",
        "ToolMetadata",
    },
    "crewai": {"tool", "BaseTool", "Tool"},
    "autogen": {"FunctionTool", "register_for_llm", "register_function"},
    "haystack": {"Tool", "ComponentTool", "PipelineTool", "tool"},
    "semantic_kernel": {
        "kernel_function",
        "KernelFunctionFromPrompt",
        "KernelFunctionFromMethod",
        "add_function",
        "create_from_method",
    },
    "camel": {"FunctionTool", "tool"},
    "metagpt": {"register_tool"},
    "autogpt": {"command", "Command"},
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
    "semantic_kernel": {
        "ChatCompletionAgent": ({"plugins"}, set()),
        "OpenAIAssistantAgent": ({"plugins"}, set()),
    },
    "camel": {"ChatAgent": ({"tools", "external_tools"}, set())},
    "metagpt": {
        "DataInterpreter": ({"tools"}, set()),
        "Role": ({"actions"}, set()),
    },
    "autogpt": {},
}


@dataclass(frozen=True)
class ToolDescriptionRecord:
    framework: str
    repo: str
    record_kind: str
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
    official_qualname: str
    bound_context: str


@dataclass(frozen=True)
class AgentBindingRecord:
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


def unparse(node: ast.AST | None, limit: int = 1200) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)[:limit]
    except Exception:
        return ast.dump(node)[:limit]


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


def is_root_module(module: str | None, roots: tuple[str, ...]) -> bool:
    return bool(module) and any(module == root or module.startswith(root + ".") for root in roots)


def first_doc_paragraph(doc: str) -> str:
    if not doc:
        return ""
    parts = doc.strip().split("\n\n", 1)
    return " ".join(parts[0].strip().split())


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
        self.class_docs: dict[str, str] = {}
        self.tool_vars: dict[str, ToolDescriptionRecord] = {}
        self.list_vars: dict[str, list[ast.AST]] = {}
        self.records: list[ToolDescriptionRecord] = []
        self.bindings: list[AgentBindingRecord] = []
        self.decorator_call_positions: set[tuple[int, int]] = set()

    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:320]
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
            if is_root_module(alias.name, self.roots):
                local = alias.asname or alias.name.split(".")[0]
                self.official_module_aliases.add(local)
                self.alias_to_qualified[local] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if is_root_module(module, self.roots):
            for alias in node.names:
                local = alias.asname or alias.name
                self.alias_to_qualified[local] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def official_qualname(self, full: str | None, last: str | None) -> str:
        if not last:
            return ""
        if last in self.alias_to_qualified:
            return self.alias_to_qualified[last]
        if full:
            parts = full.split(".")
            if parts:
                root = parts[0]
                if root in self.alias_to_qualified:
                    return ".".join([self.alias_to_qualified[root], *parts[1:]])
                if root in self.official_module_aliases:
                    return full
        return ""

    def official_expr_qualname(self, node: ast.AST) -> str:
        full, last = call_name(node)
        return self.official_qualname(full, last)

    def is_confirmed_official_call(self, full: str | None, last: str | None) -> bool:
        return bool(self.official_qualname(full, last))

    def function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args: list[str] = []
        all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        for arg in all_args:
            ann = f": {unparse(arg.annotation, 200)}" if arg.annotation else ""
            args.append(f"{arg.arg}{ann}")
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        ret = f" -> {unparse(node.returns, 200)}" if node.returns else ""
        return f"{node.name}({', '.join(args)}){ret}"

    def make_record(
        self,
        *,
        record_kind: str,
        api: str,
        line: int,
        name: str = "",
        description: str = "",
        signature: str = "",
        input_schema: str = "",
        output_schema: str = "",
        source_expr: str = "",
        official_qualname: str = "",
        bound_context: str = "",
    ) -> ToolDescriptionRecord:
        return ToolDescriptionRecord(
            framework=self.framework,
            repo=self.repo,
            record_kind=record_kind,
            api=api,
            file=self.rel_path,
            line=line,
            code=self.line_text(line),
            name=name,
            description=description,
            signature=signature,
            input_schema=input_schema,
            output_schema=output_schema,
            source_expr=source_expr,
            official_qualname=official_qualname,
            bound_context=bound_context,
        )

    def decorator_info(self, deco: ast.AST) -> tuple[str | None, ast.Call | None, str]:
        if isinstance(deco, ast.Call):
            full, last = call_name(deco.func)
            return last, deco, self.official_qualname(full, last)
        full, last = call_name(deco)
        return last, None, self.official_qualname(full, last)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_mock_like_name(node.name):
            return
        doc = ast.get_docstring(node, clean=False) or ""
        self.function_docs[node.name] = doc
        self.function_sigs[node.name] = self.function_signature(node)
        self.function_returns[node.name] = unparse(node.returns, 200)

        for deco in node.decorator_list:
            api, call, qualname = self.decorator_info(deco)
            if not qualname:
                continue
            if api in TOOL_DEFINITION_APIS.get(self.framework, set()):
                if call is not None:
                    self.decorator_call_positions.add((getattr(call, "lineno", -1), getattr(call, "col_offset", -1)))
                record = self.record_from_decorated_function(api or "", node, call, qualname)
                self.records.append(record)
                self.tool_vars[node.name] = record
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def record_from_decorated_function(
        self,
        api: str,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        deco_call: ast.Call | None,
        qualname: str,
    ) -> ToolDescriptionRecord:
        name = fn.name
        description = self.function_docs.get(fn.name, "")
        input_schema = ""
        if deco_call:
            if deco_call.args:
                first = self.static_string(deco_call.args[0])
                if first:
                    name = first
            if self.framework == "autogpt" and api == "command":
                if len(deco_call.args) >= 2:
                    description = self.static_string(deco_call.args[1]) or description
                if len(deco_call.args) >= 3:
                    input_schema = self.schema_summary(deco_call.args[2])
            for kw in deco_call.keywords:
                if kw.arg in {"name", "tool_name"}:
                    name = self.static_string(kw.value) or name
                elif kw.arg in {"description", "desc"}:
                    description = self.static_string(kw.value) or description
                elif kw.arg in {"parameters", "args_schema", "input_schema"}:
                    input_schema = self.schema_summary(kw.value)
        if self.framework == "autogpt" and api == "command" and not description:
            description = first_doc_paragraph(self.function_docs.get(fn.name, ""))
        return self.make_record(
            record_kind="custom_tool_definition",
            api=api,
            line=fn.lineno,
            name=name,
            description=description,
            signature=self.function_sigs.get(fn.name, ""),
            input_schema=input_schema,
            output_schema=self.function_returns.get(fn.name, ""),
            source_expr=fn.name,
            official_qualname=qualname,
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_mock_like_name(node.name):
            return
        self.class_docs[node.name] = ast.get_docstring(node, clean=False) or ""

        register_tool_qualname = ""
        for deco in node.decorator_list:
            api, _call, qualname = self.decorator_info(deco)
            if api == "register_tool" and qualname:
                register_tool_qualname = qualname
            if qualname:
                self.decorator_call_positions.add((getattr(deco, "lineno", -1), getattr(deco, "col_offset", -1)))

        base_qualnames = {
            self.official_expr_qualname(base)
            for base in node.bases
            if isinstance(base, (ast.Name, ast.Attribute))
        }
        base_names = {call_name(base)[1] for base in node.bases if isinstance(base, (ast.Name, ast.Attribute))}
        official_basetool = any(q.endswith(".BaseTool") for q in base_qualnames if q)
        if official_basetool or register_tool_qualname:
            record = self.record_from_tool_class(node, register_tool_qualname)
            self.records.append(record)
            self.tool_vars[node.name] = record

        official_action = self.framework == "metagpt" and (
            any(q.endswith(".Action") for q in base_qualnames if q) or "Action" in self.alias_to_qualified
        )
        if "Action" in base_names and official_action:
            record = self.record_from_action_class(node)
            self.records.append(record)
            self.tool_vars[node.name] = record

        self.generic_visit(node)

    def record_from_tool_class(self, node: ast.ClassDef, register_tool_qualname: str = "") -> ToolDescriptionRecord:
        name = node.name
        description = self.class_docs.get(node.name, "")
        input_schema = ""
        signature = ""
        output_schema = ""
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name in {
                "_run",
                "_arun",
                "run",
                "__call__",
            }:
                signature = self.function_signature(stmt)
                output_schema = unparse(stmt.returns, 200)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "name":
                            name = self.static_string(stmt.value) or name
                        elif target.id in {"description", "desc"}:
                            description = self.static_string(stmt.value) or description
                        elif target.id in {"args_schema", "input_schema", "parameters"}:
                            input_schema = self.schema_summary(stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id == "name":
                    name = self.static_string(stmt.value) or name
                elif stmt.target.id in {"description", "desc"}:
                    description = self.static_string(stmt.value) or description
                elif stmt.target.id in {"args_schema", "input_schema", "parameters"}:
                    input_schema = self.schema_summary(stmt.value)
        return self.make_record(
            record_kind="custom_tool_definition",
            api="register_tool" if register_tool_qualname else "BaseTool",
            line=node.lineno,
            name=name,
            description=description,
            signature=signature,
            input_schema=input_schema,
            output_schema=output_schema,
            source_expr=node.name,
            official_qualname=register_tool_qualname,
        )

    def record_from_action_class(self, node: ast.ClassDef) -> ToolDescriptionRecord:
        name = node.name
        description = ""
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "name":
                            name = self.static_string(stmt.value) or name
                        elif target.id == "desc":
                            description = self.static_string(stmt.value) or description
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id == "name":
                    name = self.static_string(stmt.value) or name
                elif stmt.target.id == "desc":
                    description = self.static_string(stmt.value) or description
        return self.make_record(
            record_kind="metagpt_action_description",
            api="Action",
            line=node.lineno,
            name=name,
            description=description or self.class_docs.get(node.name, ""),
            source_expr=node.name,
            official_qualname="metagpt.actions.Action",
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.handle_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.handle_assignment([node.target], node.value)
        self.generic_visit(node)

    def handle_assignment(self, targets: list[ast.AST], value_node: ast.AST | None) -> None:
        value = self.static_string(value_node)
        record = self.extract_tool_record_from_call(value_node)
        elts = self.collection_elts(value_node)
        for target in targets:
            if isinstance(target, ast.Name):
                if value is not None:
                    self.string_vars[target.id] = value
                if record is not None:
                    self.tool_vars[target.id] = record
                if elts is not None:
                    self.list_vars[target.id] = elts

    def visit_Call(self, node: ast.Call) -> None:
        if (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)) in self.decorator_call_positions:
            self.generic_visit(node)
            return
        record = self.extract_tool_record_from_call(node)
        if record is not None:
            self.records.append(record)

        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last in AGENT_TOOL_SPECS.get(self.framework, {}):
            self.add_agent_binding_record(last or "", node)
        self.generic_visit(node)

    def extract_tool_record_from_call(self, node: ast.AST | None) -> ToolDescriptionRecord | None:
        if not isinstance(node, ast.Call):
            return None
        full, last = call_name(node.func)
        qualname = self.official_qualname(full, last)
        if not qualname or last not in TOOL_DEFINITION_APIS.get(self.framework, set()):
            return None
        if not self.is_allowed_tool_definition_qualname(qualname, last or ""):
            return None

        name = ""
        description = ""
        input_schema = ""
        output_schema = ""
        signature = ""
        source_expr = unparse(node)
        record_kind = "custom_tool_definition"

        fn_name = self.function_arg_name(node)
        if fn_name:
            name = fn_name
            description = self.function_docs.get(fn_name, "")
            signature = self.function_sigs.get(fn_name, "")
            output_schema = self.function_returns.get(fn_name, "")

        if last in {"Tool", "StructuredTool", "Command"} and node.args:
            first = self.static_string(node.args[0])
            if first:
                name = first
        if self.framework == "autogpt" and last == "command":
            if len(node.args) >= 1:
                name = self.static_string(node.args[0]) or name
            if len(node.args) >= 2:
                description = self.static_string(node.args[1]) or description
            if len(node.args) >= 3:
                input_schema = input_schema or self.schema_summary(node.args[2])
        if last == "Tool" and len(node.args) >= 2:
            second = self.static_string(node.args[1])
            if second and not description:
                description = second
        if last == "ToolMetadata" and node.args:
            first = self.static_string(node.args[0])
            if first and not description:
                description = first

        for kw in node.keywords:
            if kw.arg in {"name", "tool_name", "function_name"}:
                name = self.static_string(kw.value) or name
            elif kw.arg in {"description", "desc"}:
                description = self.static_string(kw.value) or description
            elif kw.arg in {"args_schema", "input_schema", "parameters", "args", "schema", "fn_schema"}:
                input_schema = self.schema_summary(kw.value)
            elif kw.arg in {"return_schema", "output_schema", "response_schema"}:
                output_schema = self.schema_summary(kw.value)
            elif kw.arg in {"func", "fn", "function", "method"} and isinstance(kw.value, ast.Name):
                fn = kw.value.id
                name = name or fn
                description = description or self.function_docs.get(fn, "")
                signature = signature or self.function_sigs.get(fn, "")
                output_schema = output_schema or self.function_returns.get(fn, "")
            elif kw.arg == "openai_tool_schema":
                name2, desc2, params = self.parse_openai_tool_schema(kw.value)
                name = name or name2
                description = description or desc2
                input_schema = input_schema or params

        if self.framework == "llamaindex" and last == "from_defaults":
            api = full or last
        else:
            api = last or ""

        if not any([name, description, signature, input_schema, output_schema]):
            record_kind = "official_tool_reference"
        return self.make_record(
            record_kind=record_kind,
            api=api,
            line=getattr(node, "lineno", 0),
            name=name,
            description=description,
            signature=signature,
            input_schema=input_schema,
            output_schema=output_schema,
            source_expr=source_expr,
            official_qualname=qualname,
        )

    def is_allowed_tool_definition_qualname(self, qualname: str, last: str) -> bool:
        if self.framework == "llamaindex" and last == "from_defaults":
            return any(
                token in qualname
                for token in (
                    "FunctionTool.from_defaults",
                    "QueryEngineTool.from_defaults",
                    "RetrieverTool.from_defaults",
                    "LoadAndSearchToolSpec.from_defaults",
                )
            )
        return True

    def parse_openai_tool_schema(self, node: ast.AST) -> tuple[str, str, str]:
        if not isinstance(node, ast.Dict):
            return "", "", unparse(node)
        try:
            value = ast.literal_eval(node)
        except Exception:
            return "", "", unparse(node)
        function = value.get("function", value) if isinstance(value, dict) else {}
        if not isinstance(function, dict):
            return "", "", unparse(node)
        name = function.get("name", "") if isinstance(function.get("name", ""), str) else ""
        description = function.get("description", "") if isinstance(function.get("description", ""), str) else ""
        parameters = function.get("parameters", "")
        params_text = json.dumps(parameters, ensure_ascii=False)[:1200] if parameters else ""
        return name, description, params_text

    def schema_summary(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Dict):
            try:
                return json.dumps(ast.literal_eval(node), ensure_ascii=False)[:1200]
            except Exception:
                return unparse(node)
        return unparse(node)

    def function_arg_name(self, node: ast.Call) -> str:
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
        return ""

    def collection_elts(self, node: ast.AST | None) -> list[ast.AST] | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return list(node.elts)
        if isinstance(node, ast.Name):
            return self.list_vars.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.collection_elts(node.left)
            right = self.collection_elts(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def add_agent_binding_record(self, api: str, node: ast.Call) -> None:
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
            elts = self.collection_elts(candidate)
            if elts is None:
                single = self.tool_info_from_expr(candidate, api)
                if single is not None:
                    tools.append(single)
                continue
            for elt in elts:
                info = self.tool_info_from_expr(elt, api)
                if info is not None:
                    tools.append(info)

        if not tools:
            return
        self.bindings.append(
            AgentBindingRecord(
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

    def tool_info_from_expr(self, node: ast.AST, bound_context: str) -> dict[str, str] | None:
        expr = unparse(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {
                "expr": expr,
                "record_kind": "registry_tool_reference",
                "name": node.value,
                "description": "",
                "signature": "",
                "input_schema": "",
                "output_schema": "",
                "official_qualname": "",
            }

        record = None
        if isinstance(node, ast.Name):
            record = self.tool_vars.get(node.id)
            if record is None and node.id in self.function_docs:
                record = self.make_record(
                    record_kind="agent_bound_function",
                    api="plain_function_bound_tool",
                    line=getattr(node, "lineno", 0),
                    name=node.id,
                    description=self.function_docs.get(node.id, ""),
                    signature=self.function_sigs.get(node.id, ""),
                    output_schema=self.function_returns.get(node.id, ""),
                    source_expr=node.id,
                    bound_context=bound_context,
                )
                self.records.append(record)
        elif isinstance(node, ast.Call):
            record = self.extract_tool_record_from_call(node)
            if record is None:
                full, last = call_name(node.func)
                qualname = self.official_qualname(full, last)
                if qualname:
                    record = self.make_record(
                        record_kind="official_tool_reference",
                        api=last or "",
                        line=getattr(node, "lineno", 0),
                        name=last or "",
                        source_expr=expr,
                        official_qualname=qualname,
                        bound_context=bound_context,
                    )
                    self.records.append(record)
        else:
            full, last = call_name(node)
            qualname = self.official_qualname(full, last)
            if qualname:
                record = self.make_record(
                    record_kind="official_tool_reference",
                    api=last or "",
                    line=getattr(node, "lineno", 0),
                    name=last or "",
                    source_expr=expr,
                    official_qualname=qualname,
                    bound_context=bound_context,
                )
                self.records.append(record)

        if record is None:
            return None
        return {
            "expr": expr,
            "record_kind": record.record_kind,
            "name": record.name,
            "description": record.description,
            "signature": record.signature,
            "input_schema": record.input_schema,
            "output_schema": record.output_schema,
            "official_qualname": record.official_qualname,
        }


def analyze_file(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[ToolDescriptionRecord], list[AgentBindingRecord], str | None]:
    source = read_text(path)
    if source is None:
        return [], [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], [], "syntax_error"
    analyzer = FileAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    return analyzer.records, analyzer.bindings, None


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths = args
    repo_dir = Path(repo_path_str)
    records: list[ToolDescriptionRecord] = []
    bindings: list[AgentBindingRecord] = []
    py_count = 0
    read_errors = 0
    syntax_errors = 0
    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_count += 1
        try:
            file_records, file_bindings, error = analyze_file(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        records.extend(file_records)
        bindings.extend(file_bindings)
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
        "records": records,
        "bindings": bindings,
    }


def list_repos(base: Path, frameworks: list[str], limit: int = 0) -> list[tuple[str, str, str]]:
    repos: list[tuple[str, str, str]] = []
    for framework in frameworks:
        fw_dir = base / framework
        if not fw_dir.is_dir():
            raise FileNotFoundError(f"Missing framework directory: {fw_dir}")
        entries = [entry for entry in sorted(fw_dir.iterdir(), key=lambda p: p.name) if entry.is_dir()]
        if limit:
            entries = entries[:limit]
        for entry in entries:
            repos.append((framework, entry.name, str(entry)))
    return repos


def dedupe_records(records: Iterable[ToolDescriptionRecord]) -> list[ToolDescriptionRecord]:
    seen: set[tuple[str, str, str, str, int, str, str, str]] = set()
    out: list[ToolDescriptionRecord] = []
    for record in records:
        key = (
            record.framework,
            record.repo,
            record.file,
            record.record_kind,
            record.line,
            record.api,
            record.name,
            record.source_expr,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def write_outputs(results: list[dict], out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "tool_description_records.jsonl"
    official_path = out_dir / "official_tool_references.jsonl"
    bindings_path = out_dir / "agent_tool_bindings.jsonl"
    repo_csv = out_dir / "tool_description_repo_summary.csv"
    fw_csv = out_dir / "tool_description_summary_by_framework.csv"
    meta_path = out_dir / "metadata.json"

    all_records = dedupe_records(record for row in results for record in row["records"])
    with records_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with official_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            if record.record_kind == "official_tool_reference":
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with bindings_path.open("w", encoding="utf-8") as f:
        for row in results:
            for record in row["bindings"]:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "framework",
            "repo",
            "py_files",
            "read_errors",
            "syntax_errors",
            "record_count",
            "custom_tool_definition",
            "agent_bound_function",
            "official_tool_reference",
            "registry_tool_reference",
            "metagpt_action_description",
            "binding_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        records_by_repo: dict[tuple[str, str], list[ToolDescriptionRecord]] = {}
        for record in all_records:
            records_by_repo.setdefault((record.framework, record.repo), []).append(record)
        for row in results:
            rows = records_by_repo.get((row["framework"], row["repo"]), [])
            counts = {kind: sum(1 for record in rows if record.record_kind == kind) for kind in fieldnames[6:11]}
            writer.writerow(
                {
                    "framework": row["framework"],
                    "repo": row["repo"],
                    "py_files": row["py_files"],
                    "read_errors": row["read_errors"],
                    "syntax_errors": row["syntax_errors"],
                    "record_count": len(rows),
                    **counts,
                    "binding_count": len(row["bindings"]),
                }
            )

    summary_rows = []
    frameworks = sorted({row["framework"] for row in results})
    for fw in frameworks:
        rows = [row for row in results if row["framework"] == fw]
        fw_records = [record for record in all_records if record.framework == fw]
        fw_bindings = [binding for row in rows for binding in row["bindings"]]
        summary_rows.append(
            {
                "framework": fw,
                "population_repos": len(rows),
                "repos_with_records": len({record.repo for record in fw_records}),
                "repos_with_custom_or_bound_tools": len(
                    {
                        record.repo
                        for record in fw_records
                        if record.record_kind in {"custom_tool_definition", "agent_bound_function"}
                    }
                ),
                "repos_with_official_refs": len(
                    {record.repo for record in fw_records if record.record_kind == "official_tool_reference"}
                ),
                "repos_with_bindings": len({binding.repo for binding in fw_bindings}),
                "record_count": len(fw_records),
                "custom_tool_definition": sum(1 for record in fw_records if record.record_kind == "custom_tool_definition"),
                "agent_bound_function": sum(1 for record in fw_records if record.record_kind == "agent_bound_function"),
                "official_tool_reference": sum(1 for record in fw_records if record.record_kind == "official_tool_reference"),
                "registry_tool_reference": sum(1 for record in fw_records if record.record_kind == "registry_tool_reference"),
                "metagpt_action_description": sum(
                    1 for record in fw_records if record.record_kind == "metagpt_action_description"
                ),
                "binding_count": len(fw_bindings),
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

    metadata = {
        **metadata,
        "output_files": {
            "records": str(records_path),
            "official_references": str(official_path),
            "bindings": str(bindings_path),
            "repo_summary": str(repo_csv),
            "framework_summary": str(fw_csv),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp4_bp5_tool_description_dataset_v2")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-non-impl-paths", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
        raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, args.limit)
    print(f"Analyzing {len(repos)} repositories: {', '.join(frameworks)}", file=sys.stderr)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(
        results,
        out_dir,
        {
            "script": "extract_tool_description_dataset_v2.py",
            "base": str(base),
            "frameworks": frameworks,
            "repo_count": len(repos),
            "exclude_non_impl_paths": args.exclude_non_impl_paths,
            "limit": args.limit,
            "workers": args.workers,
            "purpose": "Raw tool-description dataset for BP4/BP5 downstream analyses.",
        },
    )
    print(f"Wrote results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
