#!/usr/bin/env python3
"""High-precision extraction of agent and tool descriptions from downstream repos.

This extractor is intentionally stricter than the earlier BP2/BP5 raw extractors.
It scans every repository for all supported framework APIs and custom OpenAI-style
tool schemas, but it only emits natural-language evidence when the evidence is
bound to an agent/tool construction, registration, binding, or LLM call site.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from rq3_filters import iter_py_files


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

FRAMEWORK_DISPLAY = {
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "llamaindex": "LlamaIndex",
    "crewai": "CrewAI",
    "autogen": "AutoGen",
    "haystack": "Haystack",
    "semantic_kernel": "Semantic Kernel",
    "camel": "CAMEL",
    "metagpt": "MetaGPT",
    "autogpt": "AutoGPT",
    "custom": "Custom",
}

OFFICIAL_ROOTS_BY_FRAMEWORK = {
    "langchain": ("langchain", "langchain_core", "langchain_classic", "langchain_community"),
    "langgraph": ("langgraph",),
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
    },
    "langgraph": {"create_agent", "create_react_agent"},
    "llamaindex": {"FunctionAgent", "ReActAgent", "OpenAIAgent", "AgentRunner", "AgentWorkflow", "from_tools"},
    "crewai": {"Agent", "Crew"},
    "autogen": {"AssistantAgent", "UserProxyAgent", "ConversableAgent"},
    "haystack": {"Agent"},
    "semantic_kernel": {"ChatCompletionAgent", "OpenAIAssistantAgent", "AssistantAgent"},
    "camel": {"ChatAgent", "RolePlaying"},
    "metagpt": {"Role", "DataInterpreter"},
    "autogpt": set(),
}

AGENT_FIELD_NAMES = {
    "name",
    "agent_name",
    "ai_name",
    "role",
    "agent_role",
    "ai_role",
    "profile",
    "persona",
    "goal",
    "agent_goal",
    "ai_goals",
    "backstory",
    "background",
    "agent_backstory",
    "description",
    "desc",
    "handoff_description",
    "instructions",
    "instruction",
    "system_prompt",
    "system_message",
    "prompt",
    "task_prompt",
    "context",
    "constraints",
    "constraint",
    "critic_criteria",
    "state_modifier",
    "messages_modifier",
    "assistant_role_name",
    "user_role_name",
    "critic_role_name",
}

AGENT_POSITIONAL_FIELDS = {
    "AssistantAgent": {0: "name"},
    "UserProxyAgent": {0: "name"},
    "ConversableAgent": {0: "name"},
    "ChatAgent": {0: "system_message"},
    "Role": {0: "name", 1: "profile", 2: "goal", 3: "constraints"},
}

AGENT_FIELD_PRIORITY = [
    "name",
    "agent_name",
    "ai_name",
    "role",
    "agent_role",
    "ai_role",
    "profile",
    "persona",
    "goal",
    "agent_goal",
    "ai_goals",
    "backstory",
    "background",
    "agent_backstory",
    "description",
    "desc",
    "handoff_description",
    "instructions",
    "instruction",
    "system_prompt",
    "system_message",
    "prompt",
    "task_prompt",
    "context",
    "constraints",
    "constraint",
    "critic_criteria",
    "state_modifier",
    "messages_modifier",
    "assistant_role_name",
    "user_role_name",
    "critic_role_name",
]

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
    },
    "crewai": {"Agent": ({"tools"}, set()), "Crew": ({"agents"}, set())},
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

PROMPT_BUILDER_NAMES = {
    "from_messages",
    "from_template",
    "PromptTemplate",
    "ChatPromptTemplate",
    "SystemMessage",
    "SystemMessagePromptTemplate",
    "HumanMessagePromptTemplate",
}

AGENT_LIKE_RE = re.compile(r"(agent|planner|router|worker|orchestrator|executor|crew|assistant)", re.IGNORECASE)
AGENT_TEXT_SIGNAL_RE = re.compile(
    r"\b(you are|your role|your goal|as an agent|assistant|agent|persona|role|goal|backstory|constraints?)\b",
    re.IGNORECASE,
)

EXTRACTION_SIGNAL_RE = re.compile(
    r"("
    r"AssistantAgent|ConversableAgent|UserProxyAgent|ChatCompletionAgent|OpenAIAssistantAgent|"
    r"ChatAgent|RolePlaying|DataInterpreter|\bAgent\s*\(|\bCrew\s*\(|\bRole\s*\(|class\s+\w+\([^)]*Role|"
    r"@tool\b|StructuredTool|FunctionTool|ToolMetadata|QueryEngineTool|ComponentTool|PipelineTool|"
    r"@kernel_function|register_for_llm|BaseTool|"
    r"system_message|system_prompt|instructions\s*=|messages\s*=|tools\s*=|function_map|functions\s*=|plugins\s*=|"
    r"chat\.completions\.create|responses\.create"
    r")",
    re.IGNORECASE,
)

SIGNAL_FILE_NAME_RE = re.compile(r"(agent|agents|crew|tool|tools|prompt|prompts|planner|router|worker)", re.IGNORECASE)
MAX_SOURCE_BYTES = 1_000_000
FRAMEWORK_SOURCE_PARTS = {
    "autogen",
    "autogen_agentchat",
    "autogen_core",
    "autogen_ext",
    "autogpt",
    "langchain",
    "langchain_core",
    "langchain_classic",
    "langchain_community",
    "langchain_experimental",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "langchain_google_vertexai",
    "langchain_text_splitters",
    "langgraph",
    "langserve",
    "langsmith",
    "llamaindex",
    "llama_index",
    "semantic_kernel",
    "crewai",
    "camel",
    "metagpt",
    "haystack",
}

THIRD_PARTY_SOURCE_PARTS = {
    "numpy",
    "scipy",
    "pandas",
    "torch",
    "torchvision",
    "transformers",
    "sklearn",
    "pydoc_data",
    "pypdf",
    "pypdf2",
    "mmdet",
    "paddleocr",
    "parlai",
    "colossalai",
}

FRAMEWORK_INTERNAL_PARTS = {
    "agents",
    "callbacks",
    "chains",
    "chat_models",
    "document_loaders",
    "embeddings",
    "graphs",
    "memory",
    "output_parsers",
    "prompts",
    "retrievers",
    "runnables",
    "schema",
    "tools",
    "utilities",
    "utils",
    "vectorstores",
    "workers",
}

BUNDLED_FRAMEWORK_SUBPATHS = {
    ("autogen", "python", "packages"),
    ("auto-gpt", "autogpt"),
    ("auto_gpt", "autogpt"),
    ("autogpts", "autogpt", "autogpt"),
}


@dataclass(frozen=True)
class FunctionInfo:
    module: str
    name: str
    file: str
    line: int
    docstring: str
    signature: str
    return_annotation: str


@dataclass(frozen=True)
class ClassInfo:
    module: str
    name: str
    file: str
    line: int
    bases: tuple[str, ...]
    docstring: str
    attrs: dict[str, str]
    methods: dict[str, FunctionInfo]


@dataclass
class ModuleInfo:
    module: str
    path: Path
    repo_dir: Path
    source: str
    lines: list[str]
    alias_to_qualified: dict[str, str]
    official_module_aliases: dict[str, str]
    local_imports: dict[str, tuple[str, str]]
    string_vars: dict[str, str]
    dict_vars: dict[str, dict[str, Any]]
    list_vars: dict[str, list[Any]]
    prompt_vars: dict[str, str]
    functions: dict[str, FunctionInfo]
    classes: dict[str, ClassInfo]
    agent_vars: dict[str, tuple[str, str]]
    tool_vars: dict[str, "ToolRecord"]
    tool_list_vars: dict[str, list[ast.AST]]
    file_agent_fields: dict[str, str]


@dataclass(frozen=True)
class AgentRecord:
    dataset_framework: str
    detected_framework: str
    repo: str
    record_id: str
    source_kind: str
    api: str
    file: str
    line: int
    agent_name: str
    role: str
    goal: str
    backstory: str
    description: str
    instructions: str
    system_prompt: str
    constraints: str
    context: str
    content: str
    binding_target: str
    extraction_rule: str
    confidence: str


@dataclass(frozen=True)
class ToolRecord:
    dataset_framework: str
    detected_framework: str
    repo: str
    record_id: str
    source_kind: str
    api: str
    file: str
    line: int
    tool_name: str
    description: str
    signature: str
    parameters: str
    input_schema: str
    output_schema: str
    return_annotation: str
    bound_agent: str
    binding_target: str
    extraction_rule: str
    confidence: str


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def rel(path: Path, repo_dir: Path) -> str:
    return path.relative_to(repo_dir).as_posix()


def module_name_for(path: Path, repo_dir: Path) -> str:
    return ".".join(path.relative_to(repo_dir).with_suffix("").parts)


def is_bundled_framework_source(path: Path, repo_dir: Path) -> bool:
    parts = [part.lower() for part in path.relative_to(repo_dir).parts]
    if "site-packages" in parts or "dist-packages" in parts:
        return True
    if any(part in THIRD_PARTY_SOURCE_PARTS for part in parts):
        return True
    joined = "/".join(parts)
    for pattern in BUNDLED_FRAMEWORK_SUBPATHS:
        pattern_text = "/".join(pattern)
        if pattern_text in joined:
            return True
    if ("/libs/partners/" in f"/{joined}/" or "/libs/cli/" in f"/{joined}/") and any(
        part.startswith("langchain_") or part in FRAMEWORK_SOURCE_PARTS for part in parts
    ):
        return True
    for idx, part in enumerate(parts[:-1]):
        if part.startswith("auto-gpt") and "autogpt" in parts[idx + 1 :]:
            return True
        if part in {"autogpt", "camel", "metagpt"}:
            rest = parts[idx + 1 :]
            if "setup.py" in rest:
                return True
            internal_roots = {
                "agents",
                "commands",
                "config",
                "core",
                "datasets",
                "environment",
                "memory",
                "models",
                "roles",
                "schemas",
                "societies",
                "toolkits",
                "utils",
            }
            if rest and rest[0] in internal_roots:
                return True
    if "python/packages/" in joined and any(part in FRAMEWORK_SOURCE_PARTS for part in parts):
        return True
    for idx, part in enumerate(parts[:-1]):
        if part in FRAMEWORK_SOURCE_PARTS:
            rest = parts[idx + 1 :]
            if rest and rest[0] in FRAMEWORK_INTERNAL_PARTS:
                return True
            if any(parent in {"libs", "lib", "layers", "package", "packages", "python", "src"} for parent in parts[:idx]):
                return True
    vendored_parents = {"vendor", "vendors", "third_party", "third-party", "external", "deps", "submodules", "bundled"}
    if any(part in FRAMEWORK_SOURCE_PARTS and any(parent in vendored_parents for parent in parts[:idx]) for idx, part in enumerate(parts)):
        return True
    if any(part in {"src", "packages"} for part in parts):
        for idx, part in enumerate(parts[:-1]):
            if part in FRAMEWORK_SOURCE_PARTS and idx > 0 and parts[idx - 1] in {"src", "packages"}:
                return True
    return False


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


def call_name(node: ast.AST | None) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        parts = attr_chain(node)
        if parts:
            return ".".join(parts), parts[-1]
    return None, None


def unparse(node: ast.AST | None, limit: int = 1200) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)[:limit]
    except Exception:
        return ast.dump(node)[:limit]


def line_text(info: ModuleInfo, lineno: int) -> str:
    if 1 <= lineno <= len(info.lines):
        return info.lines[lineno - 1].strip()[:400]
    return ""


def official_framework_for_module(module: str | None) -> str | None:
    if not module:
        return None
    for framework, roots in OFFICIAL_ROOTS_BY_FRAMEWORK.items():
        if any(module == root or module.startswith(f"{root}.") for root in roots):
            return framework
    return None


def framework_from_qualified(qualified: str | None) -> str | None:
    if not qualified:
        return None
    return official_framework_for_module(qualified)


def resolve_import_module(current_module: str, module: str | None, level: int) -> str:
    module = module or ""
    if level <= 0:
        return module
    parts = current_module.split(".")[:-level]
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)


class RepoIndex:
    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.modules: dict[str, ModuleInfo] = {}
        self.modules_by_basename: dict[str, list[str]] = {}

    def add_module(self, info: ModuleInfo) -> None:
        self.modules[info.module] = info
        self.modules_by_basename.setdefault(info.module.split(".")[-1], []).append(info.module)

    def resolve_module_name(self, name: str) -> str | None:
        if name in self.modules:
            return name
        matches = [module for module in self.modules if module.endswith(f".{name}")]
        if len(matches) == 1:
            return matches[0]
        basename_matches = self.modules_by_basename.get(name, [])
        if len(basename_matches) == 1:
            return basename_matches[0]
        return None

    def resolve_imported_string(self, info: ModuleInfo, name: str, seen: set[tuple[str, str]] | None = None) -> str | None:
        if seen is None:
            seen = set()
        if name in info.string_vars:
            return info.string_vars[name]
        if name in info.prompt_vars:
            return info.prompt_vars[name]
        imported = info.local_imports.get(name)
        if not imported:
            return None
        module_name, imported_name = imported
        resolved = self.resolve_module_name(module_name)
        if not resolved or (resolved, imported_name) in seen:
            return None
        seen.add((resolved, imported_name))
        target = self.modules.get(resolved)
        if not target:
            return None
        return self.resolve_imported_string(target, imported_name, seen)

    def resolve_function(self, info: ModuleInfo, name: str) -> FunctionInfo | None:
        if name in info.functions:
            return info.functions[name]
        imported = info.local_imports.get(name)
        if imported:
            module_name, imported_name = imported
            target = self.modules.get(self.resolve_module_name(module_name) or "")
            if target:
                return target.functions.get(imported_name)
        return None

    def resolve_class(self, info: ModuleInfo, name: str) -> ClassInfo | None:
        if name in info.classes:
            return info.classes[name]
        imported = info.local_imports.get(name)
        if imported:
            module_name, imported_name = imported
            target = self.modules.get(self.resolve_module_name(module_name) or "")
            if target:
                return target.classes.get(imported_name)
        return None

    def resolve_tool(self, info: ModuleInfo, name: str) -> "ToolRecord" | None:
        if name in info.tool_vars:
            return info.tool_vars[name]
        imported = info.local_imports.get(name)
        if imported:
            module_name, imported_name = imported
            target = self.modules.get(self.resolve_module_name(module_name) or "")
            if target:
                return target.tool_vars.get(imported_name)
        return None


def static_string(node: ast.AST | None, info: ModuleInfo, index: RepoIndex) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        chunks: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                chunks.append(value.value)
            else:
                return None
        return "".join(chunks)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_string(node.left, info, index)
        right = static_string(node.right, info, index)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.Name):
        return index.resolve_imported_string(info, node.id)
    if isinstance(node, ast.Call):
        return prompt_builder_text(node, info, index)
    return None


def static_list(node: ast.AST | None, info: ModuleInfo, index: RepoIndex) -> list[Any] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [static_value(elt, info, index) for elt in node.elts]
    if isinstance(node, ast.Name):
        if node.id in info.list_vars:
            return info.list_vars[node.id]
        imported = info.local_imports.get(node.id)
        if imported:
            module_name, imported_name = imported
            target = index.modules.get(index.resolve_module_name(module_name) or "")
            if target:
                return target.list_vars.get(imported_name)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = static_list(node.left, info, index)
        right = static_list(node.right, info, index)
        if left is not None and right is not None:
            return left + right
    return None


def static_dict(node: ast.AST | None, info: ModuleInfo, index: RepoIndex) -> dict[str, Any] | None:
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = static_string(key_node, info, index)
            if key is None:
                continue
            result[key] = static_value(value_node, info, index)
        return result
    if isinstance(node, ast.Name):
        if node.id in info.dict_vars:
            return info.dict_vars[node.id]
        imported = info.local_imports.get(node.id)
        if imported:
            module_name, imported_name = imported
            target = index.modules.get(index.resolve_module_name(module_name) or "")
            if target:
                return target.dict_vars.get(imported_name)
    if isinstance(node, ast.Subscript):
        root = static_dict(node.value, info, index)
        key = static_string(node.slice, info, index)
        if root and key and isinstance(root.get(key), dict):
            return root[key]
    return None


def static_value(node: ast.AST | None, info: ModuleInfo, index: RepoIndex) -> Any:
    string = static_string(node, info, index)
    if string is not None:
        return string
    mapping = static_dict(node, info, index)
    if mapping is not None:
        return mapping
    values = static_list(node, info, index)
    if values is not None:
        return values
    return unparse(node)


def formatted_string_text(node: ast.JoinedStr) -> str | None:
    parts: list[str] = []
    has_text = False
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
            if re.search(r"[A-Za-z\u4e00-\u9fff]", value.value):
                has_text = True
        elif isinstance(value, ast.FormattedValue):
            parts.append("{" + unparse(value.value, 200) + "}")
        else:
            return None
    text = "".join(parts).strip()
    return text if has_text and text else None


def static_description_value(node: ast.AST | None, info: ModuleInfo, index: RepoIndex) -> Any:
    string = static_string(node, info, index)
    if string is not None:
        return string
    if isinstance(node, ast.JoinedStr):
        return formatted_string_text(node)
    mapping = static_dict(node, info, index)
    if mapping is not None:
        return mapping
    values = static_list(node, info, index)
    if values is not None:
        return values
    return None


def prompt_builder_text(node: ast.Call, info: ModuleInfo, index: RepoIndex) -> str | None:
    full, last = call_name(node.func)
    if not last:
        return None
    parts = set((full or "").split("."))
    if last not in PROMPT_BUILDER_NAMES and not (parts & PROMPT_BUILDER_NAMES):
        return None
    chunks: list[str] = []
    for arg in node.args:
        collect_prompt_strings(arg, info, index, chunks)
    for kw in node.keywords:
        collect_prompt_strings(kw.value, info, index, chunks)
    return "\n".join(chunk for chunk in chunks if chunk.strip()) or None


def collect_prompt_strings(node: ast.AST | None, info: ModuleInfo, index: RepoIndex, chunks: list[str]) -> None:
    if node is None:
        return
    value = static_string(node, info, index)
    if value is not None:
        chunks.append(value)
        return
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            collect_prompt_strings(elt, info, index, chunks)
    elif isinstance(node, ast.Dict):
        for key_node, value_node in zip(node.keys, node.values):
            collect_prompt_strings(key_node, info, index, chunks)
            collect_prompt_strings(value_node, info, index, chunks)
    elif isinstance(node, ast.Call):
        text = prompt_builder_text(node, info, index)
        if text:
            chunks.append(text)


def sequence_to_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value if isinstance(item, str))
    if isinstance(value, dict):
        return "\n".join(f"{key}: {val}" for key, val in value.items() if isinstance(val, str))
    return value if isinstance(value, str) else ""


def is_concrete_agent_text(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if re.fullmatch(r"(self|cls)\.[A-Za-z_][A-Za-z0-9_]*", text):
        return False
    if text in {"None", "[]", "{}", "''", '""'}:
        return False
    return True


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
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


class IndexBuilder(ast.NodeVisitor):
    def __init__(self, info: ModuleInfo, index: RepoIndex):
        self.info = info
        self.index = index

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            framework = official_framework_for_module(alias.name)
            if framework:
                self.info.official_module_aliases[alias.asname or alias.name.split(".")[0]] = framework
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = resolve_import_module(self.info.module, node.module, node.level)
        framework = official_framework_for_module(module)
        for alias in node.names:
            local = alias.asname or alias.name
            if framework:
                self.info.alias_to_qualified[local] = f"{module}.{alias.name}"
            else:
                self.info.local_imports[local] = (module, alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.record_assignment(target.id, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.record_assignment(node.target.id, node.value)
        self.generic_visit(node)

    def record_assignment(self, name: str, value: ast.AST | None) -> None:
        if value is None:
            return
        string = static_string(value, self.info, self.index)
        if string is not None:
            self.info.string_vars[name] = string
            if "prompt" in name.lower() or "instruction" in name.lower() or "message" in name.lower():
                self.info.prompt_vars[name] = string
            if name in {"ai_name", "ai_role", "ai_goals", "agent_name", "agent_role", "agent_goal"}:
                self.info.file_agent_fields[name] = string
            return
        mapping = static_dict(value, self.info, self.index)
        if mapping is not None:
            self.info.dict_vars[name] = mapping
            for key in {"ai_name", "ai_role", "ai_goals"}:
                if key in mapping:
                    self.info.file_agent_fields[key] = sequence_to_text(mapping[key])
            return
        values = static_list(value, self.info, self.index)
        if values is not None:
            self.info.list_vars[name] = values
            if name == "ai_goals":
                self.info.file_agent_fields[name] = sequence_to_text(values)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.info.functions[node.name] = FunctionInfo(
            module=self.info.module,
            name=node.name,
            file=rel(self.info.path, self.info.repo_dir),
            line=node.lineno,
            docstring=ast.get_docstring(node, clean=False) or "",
            signature=function_signature(node),
            return_annotation=unparse(node.returns),
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        attrs: dict[str, str] = {}
        methods: dict[str, FunctionInfo] = {}
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[stmt.name] = FunctionInfo(
                    module=self.info.module,
                    name=stmt.name,
                    file=rel(self.info.path, self.info.repo_dir),
                    line=stmt.lineno,
                    docstring=ast.get_docstring(stmt, clean=False) or "",
                    signature=function_signature(stmt),
                    return_annotation=unparse(stmt.returns),
                )
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        value = static_string(stmt.value, self.info, self.index)
                        if value is not None:
                            attrs[target.id] = value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                value = static_string(stmt.value, self.info, self.index)
                if value is not None:
                    attrs[stmt.target.id] = value
        self.info.classes[node.name] = ClassInfo(
            module=self.info.module,
            name=node.name,
            file=rel(self.info.path, self.info.repo_dir),
            line=node.lineno,
            bases=tuple(unparse(base) for base in node.bases),
            docstring=ast.get_docstring(node, clean=False) or "",
            attrs=attrs,
            methods=methods,
        )
        self.generic_visit(node)


class ToolPreIndexer(ast.NodeVisitor):
    """Populate module.tool_vars before agent binding analysis runs."""

    def __init__(self, dataset_framework: str, repo: str, info: ModuleInfo, index: RepoIndex):
        self.dataset_framework = dataset_framework
        self.repo = repo
        self.info = info
        self.index = index

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for deco in node.decorator_list:
            deco_call = deco if isinstance(deco, ast.Call) else None
            full, last = call_name(deco.func if isinstance(deco, ast.Call) else deco)
            frameworks = call_frameworks(self.info, full, last, TOOL_APIS)
            if not frameworks and last == "register_for_llm":
                frameworks = ["autogen"]
            for framework in frameworks:
                description = ast.get_docstring(node, clean=False) or ""
                name = node.name
                if deco_call:
                    for kw in deco_call.keywords:
                        if kw.arg == "name":
                            name = static_string(kw.value, self.info, self.index) or name
                        elif kw.arg == "description":
                            description = static_string(kw.value, self.info, self.index) or description
                self.info.tool_vars[node.name] = ToolRecord(
                    dataset_framework=self.dataset_framework,
                    detected_framework=framework,
                    repo=self.repo,
                    record_id="",
                    source_kind="official_tool_decorator",
                    api=last or "",
                    file=rel(self.info.path, self.info.repo_dir),
                    line=node.lineno,
                    tool_name=name,
                    description=description,
                    signature=function_signature(node),
                    parameters="",
                    input_schema="",
                    output_schema="",
                    return_annotation=unparse(node.returns),
                    bound_agent="",
                    binding_target=f"@{last}",
                    extraction_rule="official_tool_decorator_or_registration",
                    confidence="high",
                )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            record = self.tool_from_call(node.value)
            if record:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.info.tool_vars[target.id] = record
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Call):
            record = self.tool_from_call(node.value)
            if record:
                self.info.tool_vars[node.target.id] = record
        self.generic_visit(node)

    def tool_from_call(self, node: ast.Call) -> ToolRecord | None:
        full, last = call_name(node.func)
        frameworks = call_frameworks(self.info, full, last, TOOL_APIS)
        if not frameworks:
            return None
        framework = frameworks[0]
        name = ""
        description = ""
        signature = ""
        return_annotation = ""
        if node.args and isinstance(node.args[0], ast.Name):
            fn_info = self.index.resolve_function(self.info, node.args[0].id)
            if fn_info:
                name = fn_info.name
                description = fn_info.docstring
                signature = fn_info.signature
                return_annotation = fn_info.return_annotation
        for kw in node.keywords:
            if kw.arg in {"name", "tool_name"}:
                name = static_string(kw.value, self.info, self.index) or name
            elif kw.arg in {"description", "desc"}:
                description = static_string(kw.value, self.info, self.index) or description
            elif kw.arg in {"func", "fn", "function"} and isinstance(kw.value, ast.Name):
                fn_info = self.index.resolve_function(self.info, kw.value.id)
                if fn_info:
                    name = name or fn_info.name
                    description = description or fn_info.docstring
                    signature = signature or fn_info.signature
                    return_annotation = return_annotation or fn_info.return_annotation
        if not (name or description or signature):
            return None
        return ToolRecord(
            dataset_framework=self.dataset_framework,
            detected_framework=framework,
            repo=self.repo,
            record_id="",
            source_kind="official_tool_constructor",
            api=last or "",
            file=rel(self.info.path, self.info.repo_dir),
            line=getattr(node, "lineno", 0),
            tool_name=name,
            description=description,
            signature=signature,
            parameters="",
            input_schema="",
            output_schema="",
            return_annotation=return_annotation,
            bound_agent="",
            binding_target=last or "",
            extraction_rule="official_tool_constructor_fields",
            confidence="high",
        )


def make_empty_module_info(repo_dir: Path, path: Path, source: str) -> ModuleInfo:
    return ModuleInfo(
        module=module_name_for(path, repo_dir),
        path=path,
        repo_dir=repo_dir,
        source=source,
        lines=source.splitlines(),
        alias_to_qualified={},
        official_module_aliases={},
        local_imports={},
        string_vars={},
        dict_vars={},
        list_vars={},
        prompt_vars={},
        functions={},
        classes={},
        agent_vars={},
        tool_vars={},
        tool_list_vars={},
        file_agent_fields={},
    )


def call_frameworks(info: ModuleInfo, full: str | None, last: str | None, api_map: dict[str, set[str]]) -> list[str]:
    if not last:
        return []
    frameworks: list[str] = []
    first = (full or "").split(".")[0] if full else ""
    qualified = info.alias_to_qualified.get(last) or info.alias_to_qualified.get(first)
    framework = framework_from_qualified(qualified)
    if framework and last in api_map.get(framework, set()):
        frameworks.append(framework)
    if first in info.official_module_aliases:
        framework = info.official_module_aliases[first]
        if last in api_map.get(framework, set()):
            frameworks.append(framework)
    if first in info.alias_to_qualified:
        framework = framework_from_qualified(info.alias_to_qualified[first])
        if framework and last in api_map.get(framework, set()):
            frameworks.append(framework)
    return sorted(set(frameworks))


def field_text(fields: dict[str, str]) -> str:
    chunks: list[str] = []
    for key in AGENT_FIELD_PRIORITY:
        if fields.get(key):
            chunks.append(f"{key}: {fields[key]}")
    for key in sorted(fields):
        if key not in AGENT_FIELD_PRIORITY and fields[key]:
            chunks.append(f"{key}: {fields[key]}")
    return "\n".join(chunks)


def add_hash_id(prefix: str, payload: str) -> str:
    return f"{prefix}-{hashlib.sha1(payload.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


class RepoAnalyzer(ast.NodeVisitor):
    def __init__(self, dataset_framework: str, repo: str, info: ModuleInfo, index: RepoIndex):
        self.dataset_framework = dataset_framework
        self.repo = repo
        self.info = info
        self.index = index
        self.agent_records: list[AgentRecord] = []
        self.tool_records: list[ToolRecord] = []
        self.scope_stack: list[str] = []
        self.processed_call_nodes: set[int] = set()
        self.processed_tool_call_nodes: set[int] = set()
        self.processed_class_nodes: set[int] = set()

    def current_scope_is_agent_like(self) -> bool:
        target = " ".join([self.info.path.as_posix(), *self.scope_stack])
        return bool(AGENT_LIKE_RE.search(target))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope_stack.append(node.name)
        self.add_tool_from_decorators(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope_stack.append(node.name)
        self.add_base_tool_record(node)
        self.add_metagpt_role_record(node)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node.value, ast.Call):
            for target in targets:
                self.handle_agent_call(node.value, assigned_name=target)
                tool = self.extract_tool_from_call(node.value, bound_agent="")
                if tool:
                    self.add_tool(tool)
                    self.info.tool_vars[target] = tool
            if targets:
                elts = self.collection_elts(node.value)
                if elts is not None:
                    for target in targets:
                        self.info.tool_list_vars[target] = elts
        elif targets:
            elts = self.collection_elts(node.value)
            if elts is not None:
                for target in targets:
                    self.info.tool_list_vars[target] = elts
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            target = node.target.id
            if isinstance(node.value, ast.Call):
                self.handle_agent_call(node.value, assigned_name=target)
                tool = self.extract_tool_from_call(node.value, bound_agent="")
                if tool:
                    self.add_tool(tool)
                    self.info.tool_vars[target] = tool
            elts = self.collection_elts(node.value)
            if elts is not None:
                self.info.tool_list_vars[target] = elts
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.handle_agent_call(node, assigned_name="")
        if id(node) not in self.processed_tool_call_nodes:
            tool = self.extract_tool_from_call(node, bound_agent="")
            if tool:
                self.add_tool(tool)
        self.handle_custom_llm_call(node)
        self.generic_visit(node)

    def add_agent(self, record: AgentRecord) -> None:
        self.agent_records.append(record)

    def add_tool(self, record: ToolRecord) -> None:
        self.tool_records.append(record)

    def handle_agent_call(self, node: ast.Call, assigned_name: str) -> None:
        if id(node) in self.processed_call_nodes:
            return
        full, last = call_name(node.func)
        frameworks = call_frameworks(self.info, full, last, AGENT_APIS)
        if not frameworks:
            return
        self.processed_call_nodes.add(id(node))
        for framework in frameworks:
            fields = self.agent_fields_from_call(node, last or "")
            if not fields:
                continue
            content = field_text(fields)
            if not content:
                continue
            agent_name = fields.get("name") or fields.get("agent_name") or fields.get("ai_name") or assigned_name
            rid = add_hash_id("agent", f"{framework}:{self.repo}:{self.info.path}:{getattr(node, 'lineno', 0)}:{content}")
            record = AgentRecord(
                dataset_framework=self.dataset_framework,
                detected_framework=framework,
                repo=self.repo,
                record_id=rid,
                source_kind="official_agent_constructor",
                api=last or "",
                file=rel(self.info.path, self.info.repo_dir),
                line=getattr(node, "lineno", 0),
                agent_name=agent_name,
                role=fields.get("role") or fields.get("agent_role") or fields.get("ai_role") or fields.get("profile") or "",
                goal=fields.get("goal") or fields.get("agent_goal") or fields.get("ai_goals") or "",
                backstory=fields.get("backstory") or fields.get("background") or fields.get("agent_backstory") or "",
                description=fields.get("description") or fields.get("desc") or fields.get("handoff_description") or "",
                instructions=fields.get("instructions") or fields.get("instruction") or "",
                system_prompt=fields.get("system_prompt") or fields.get("system_message") or fields.get("prompt") or fields.get("task_prompt") or "",
                constraints=fields.get("constraints") or fields.get("constraint") or fields.get("critic_criteria") or "",
                context=fields.get("context") or "",
                content=content,
                binding_target=f"{last}",
                extraction_rule="official_agent_constructor_fields",
                confidence="high",
            )
            self.add_agent(record)
            if assigned_name:
                self.info.agent_vars[assigned_name] = (framework, agent_name or rid)
            self.extract_bound_tools_from_agent_call(framework, last or "", node, agent_name or assigned_name)

    def agent_fields_from_call(self, node: ast.Call, api: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for pos, field in AGENT_POSITIONAL_FIELDS.get(api, {}).items():
            if pos < len(node.args):
                value = sequence_to_text(static_description_value(node.args[pos], self.info, self.index))
                if value:
                    fields[field] = value
        for kw in node.keywords:
            if kw.arg == "config":
                cfg = static_dict(kw.value, self.info, self.index)
                if cfg:
                    for key, value in cfg.items():
                        if key in AGENT_FIELD_NAMES:
                            text = sequence_to_text(value)
                            if text:
                                fields[key] = text
                continue
            if kw.arg not in AGENT_FIELD_NAMES:
                continue
            value = sequence_to_text(static_description_value(kw.value, self.info, self.index))
            if value:
                fields[kw.arg] = value
        return fields

    def add_metagpt_role_record(self, node: ast.ClassDef) -> None:
        if id(node) in self.processed_class_nodes:
            return
        imported_metagpt = any(framework == "metagpt" for framework in self.info.official_module_aliases.values()) or any(
            framework_from_qualified(q) == "metagpt" for q in self.info.alias_to_qualified.values()
        )
        if not imported_metagpt:
            return
        fields: dict[str, str] = {"name": node.name}
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Call):
                        _, last = call_name(child.func)
                        if last == "__init__":
                            for kw in child.keywords:
                                if kw.arg in AGENT_FIELD_NAMES:
                                    value = sequence_to_text(static_description_value(kw.value, self.info, self.index))
                                    if value:
                                        fields[kw.arg] = value
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Attribute) and target.attr in AGENT_FIELD_NAMES:
                                value = sequence_to_text(static_description_value(child.value, self.info, self.index))
                                if value:
                                    fields[target.attr] = value
        if len(fields) <= 1:
            return
        self.processed_class_nodes.add(id(node))
        content = field_text(fields)
        rid = add_hash_id("agent", f"metagpt:{self.repo}:{self.info.path}:{node.lineno}:{content}")
        self.add_agent(
            AgentRecord(
                dataset_framework=self.dataset_framework,
                detected_framework="metagpt",
                repo=self.repo,
                record_id=rid,
                source_kind="official_agent_role_class",
                api="RoleClass",
                file=rel(self.info.path, self.info.repo_dir),
                line=node.lineno,
                agent_name=fields.get("name", ""),
                role=fields.get("profile", ""),
                goal=fields.get("goal", ""),
                backstory=fields.get("backstory", ""),
                description=fields.get("description", ""),
                instructions=fields.get("instructions", ""),
                system_prompt=fields.get("prompt", ""),
                constraints=fields.get("constraints", ""),
                context=fields.get("context", ""),
                content=content,
                binding_target="Role.__init__",
                extraction_rule="metagpt_role_subclass_fields",
                confidence="high",
            )
        )

    def add_autogpt_file_record(self) -> None:
        if not self.info.file_agent_fields:
            return
        if not ({"ai_name", "ai_role", "ai_goals"} & set(self.info.file_agent_fields)):
            return
        concrete_fields = {key: value for key, value in self.info.file_agent_fields.items() if is_concrete_agent_text(value)}
        if not concrete_fields:
            return
        content = field_text(concrete_fields)
        if not content:
            return
        rid = add_hash_id("agent", f"autogpt:{self.repo}:{self.info.path}:0:{content}")
        self.add_agent(
            AgentRecord(
                dataset_framework=self.dataset_framework,
                detected_framework="autogpt",
                repo=self.repo,
                record_id=rid,
                source_kind="autogpt_ai_profile",
                api="AutoGPTAISpec",
                file=rel(self.info.path, self.info.repo_dir),
                line=0,
                agent_name=concrete_fields.get("ai_name", ""),
                role=concrete_fields.get("ai_role", ""),
                goal=concrete_fields.get("ai_goals", ""),
                backstory="",
                description="",
                instructions="",
                system_prompt="",
                constraints="",
                context="",
                content=content,
                binding_target="ai_name/ai_role/ai_goals",
                extraction_rule="autogpt_profile_fields",
                confidence="medium",
            )
        )

    def handle_custom_llm_call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        full_lower = (full or "").lower()
        if not (
            full_lower.endswith("chat.completions.create")
            or full_lower.endswith("responses.create")
            or "chatcompletion.create" in full_lower
        ):
            return
        self.extract_custom_agent_prompt_from_llm_call(node, full or last or "llm_call")
        self.extract_openai_tool_schemas(node, bound_agent="OpenAI")

    def extract_custom_agent_prompt_from_llm_call(self, node: ast.Call, api: str) -> None:
        if not self.current_scope_is_agent_like():
            return
        fields: dict[str, str] = {}
        for kw in node.keywords:
            if kw.arg in {"system_prompt", "instructions"}:
                text = sequence_to_text(static_value(kw.value, self.info, self.index))
                if text and AGENT_TEXT_SIGNAL_RE.search(text):
                    fields[kw.arg] = text
            elif kw.arg == "messages":
                text = self.extract_system_messages(kw.value)
                if text and AGENT_TEXT_SIGNAL_RE.search(text):
                    fields["system_message"] = text
        content = field_text(fields)
        if not content:
            return
        rid = add_hash_id("agent", f"custom:{self.repo}:{self.info.path}:{getattr(node, 'lineno', 0)}:{content}")
        self.add_agent(
            AgentRecord(
                dataset_framework=self.dataset_framework,
                detected_framework="custom",
                repo=self.repo,
                record_id=rid,
                source_kind="custom_llm_agent_prompt",
                api=api,
                file=rel(self.info.path, self.info.repo_dir),
                line=getattr(node, "lineno", 0),
                agent_name="",
                role="",
                goal="",
                backstory="",
                description="",
                instructions=fields.get("instructions", ""),
                system_prompt=fields.get("system_prompt") or fields.get("system_message", ""),
                constraints="",
                context="",
                content=content,
                binding_target=f"{api}.messages",
                extraction_rule="agent_like_llm_system_prompt_binding",
                confidence="medium",
            )
        )

    def extract_system_messages(self, node: ast.AST) -> str:
        value = static_value(node, self.info, self.index)
        chunks: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("role") in {"system", "developer"}:
                    chunks.append(sequence_to_text(item.get("content")))
                elif isinstance(item, str):
                    chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk)

    def add_tool_from_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for deco in node.decorator_list:
            deco_call = deco if isinstance(deco, ast.Call) else None
            if deco_call is not None:
                self.processed_tool_call_nodes.add(id(deco_call))
            full, last = call_name(deco.func if isinstance(deco, ast.Call) else deco)
            frameworks = call_frameworks(self.info, full, last, TOOL_APIS)
            if not frameworks and last == "register_for_llm":
                frameworks = ["autogen"]
            for framework in frameworks:
                description = ast.get_docstring(node, clean=False) or ""
                name = node.name
                if deco_call:
                    for kw in deco_call.keywords:
                        if kw.arg == "name":
                            name = static_string(kw.value, self.info, self.index) or name
                        elif kw.arg == "description":
                            description = static_string(kw.value, self.info, self.index) or description
                rid = add_hash_id("tool", f"{framework}:{self.repo}:{self.info.path}:{node.lineno}:{name}:{description}")
                record = ToolRecord(
                    dataset_framework=self.dataset_framework,
                    detected_framework=framework,
                    repo=self.repo,
                    record_id=rid,
                    source_kind="official_tool_decorator",
                    api=last or "",
                    file=rel(self.info.path, self.info.repo_dir),
                    line=node.lineno,
                    tool_name=name,
                    description=description,
                    signature=function_signature(node),
                    parameters="",
                    input_schema="",
                    output_schema="",
                    return_annotation=unparse(node.returns),
                    bound_agent="",
                    binding_target=f"@{last}",
                    extraction_rule="official_tool_decorator_or_registration",
                    confidence="high",
                )
                self.add_tool(record)
                self.info.tool_vars[node.name] = record

    def add_base_tool_record(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            _, base_last = call_name(base)
            if base_last != "BaseTool":
                continue
            qualified = self.info.alias_to_qualified.get("BaseTool")
            framework = framework_from_qualified(qualified)
            if framework not in {"langchain", "langgraph", "crewai"}:
                continue
            cls = self.info.classes.get(node.name)
            if not cls:
                continue
            name = cls.attrs.get("name") or node.name
            description = cls.attrs.get("description") or cls.docstring
            input_schema = cls.attrs.get("args_schema") or cls.attrs.get("input_schema") or ""
            method = cls.methods.get("_run") or cls.methods.get("_arun") or cls.methods.get("run")
            rid = add_hash_id("tool", f"{framework}:{self.repo}:{self.info.path}:{node.lineno}:{name}:{description}")
            self.add_tool(
                ToolRecord(
                    dataset_framework=self.dataset_framework,
                    detected_framework=framework,
                    repo=self.repo,
                    record_id=rid,
                    source_kind="official_tool_subclass",
                    api="BaseTool",
                    file=rel(self.info.path, self.info.repo_dir),
                    line=node.lineno,
                    tool_name=name,
                    description=description,
                    signature=method.signature if method else "",
                    parameters="",
                    input_schema=input_schema,
                    output_schema="",
                    return_annotation=method.return_annotation if method else "",
                    bound_agent="",
                    binding_target="BaseTool subclass",
                    extraction_rule="framework_confirmed_basetool_subclass",
                    confidence="high",
                )
            )

    def extract_tool_from_call(self, node: ast.Call, bound_agent: str) -> ToolRecord | None:
        full, last = call_name(node.func)
        frameworks = call_frameworks(self.info, full, last, TOOL_APIS)
        if not frameworks:
            return None
        framework = frameworks[0]
        name = ""
        description = ""
        signature = ""
        parameters = ""
        input_schema = ""
        output_schema = ""
        return_annotation = ""
        fn = self.function_arg_name(node)
        if fn:
            fn_info = self.index.resolve_function(self.info, fn)
            if fn_info:
                name = fn_info.name
                description = fn_info.docstring
                signature = fn_info.signature
                return_annotation = fn_info.return_annotation
        for kw in node.keywords:
            if kw.arg in {"name", "tool_name"}:
                name = static_string(kw.value, self.info, self.index) or name
            elif kw.arg in {"description", "desc"}:
                description = static_string(kw.value, self.info, self.index) or description
            elif kw.arg in {"args_schema", "input_schema", "parameters", "args", "schema"}:
                input_schema = unparse(kw.value)
                parameters = input_schema
            elif kw.arg in {"return_schema", "output_schema", "response_schema"}:
                output_schema = unparse(kw.value)
            elif kw.arg in {"func", "fn", "function"} and isinstance(kw.value, ast.Name):
                fn_info = self.index.resolve_function(self.info, kw.value.id)
                if fn_info:
                    name = name or fn_info.name
                    description = description or fn_info.docstring
                    signature = signature or fn_info.signature
                    return_annotation = return_annotation or fn_info.return_annotation
        if not (name or description or signature or input_schema or output_schema):
            return None
        rid = add_hash_id("tool", f"{framework}:{self.repo}:{self.info.path}:{getattr(node, 'lineno', 0)}:{name}:{description}:{bound_agent}")
        return ToolRecord(
            dataset_framework=self.dataset_framework,
            detected_framework=framework,
            repo=self.repo,
            record_id=rid,
            source_kind="official_tool_constructor",
            api=last or "",
            file=rel(self.info.path, self.info.repo_dir),
            line=getattr(node, "lineno", 0),
            tool_name=name,
            description=description,
            signature=signature,
            parameters=parameters,
            input_schema=input_schema,
            output_schema=output_schema,
            return_annotation=return_annotation,
            bound_agent=bound_agent,
            binding_target=last or "",
            extraction_rule="official_tool_constructor_fields",
            confidence="high",
        )

    def function_arg_name(self, node: ast.Call) -> str:
        for arg in node.args:
            if isinstance(arg, ast.Name):
                return arg.id
        return ""

    def collection_elts(self, node: ast.AST | None) -> list[ast.AST] | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return list(node.elts)
        if isinstance(node, ast.Name):
            if node.id in self.info.tool_list_vars:
                return self.info.tool_list_vars[node.id]
            value = self.info.list_vars.get(node.id)
            if isinstance(value, list):
                return []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.collection_elts(node.left)
            right = self.collection_elts(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def extract_bound_tools_from_agent_call(self, framework: str, api: str, node: ast.Call, agent_name: str) -> None:
        specs = AGENT_TOOL_SPECS.get(framework, {}).get(api)
        if not specs:
            return
        keywords, positions = specs
        candidates: list[ast.AST] = []
        for pos in positions:
            if pos < len(node.args):
                candidates.append(node.args[pos])
        for kw in node.keywords:
            if kw.arg in keywords:
                candidates.append(kw.value)
        for candidate in candidates:
            for elt in self.collection_elts(candidate) or []:
                record = self.tool_from_bound_expr(elt, agent_name)
                if record:
                    self.add_tool(record)

    def tool_from_bound_expr(self, node: ast.AST, agent_name: str) -> ToolRecord | None:
        if isinstance(node, ast.Name):
            resolved_tool = self.index.resolve_tool(self.info, node.id)
            if resolved_tool:
                return replace(resolved_tool, bound_agent=agent_name, binding_target=f"agent_tools:{agent_name}")
            fn_info = self.index.resolve_function(self.info, node.id)
            if fn_info and (fn_info.docstring or fn_info.signature):
                rid = add_hash_id("tool", f"custom:{self.repo}:{fn_info.file}:{fn_info.line}:{fn_info.name}:{agent_name}")
                return ToolRecord(
                    dataset_framework=self.dataset_framework,
                    detected_framework="custom",
                    repo=self.repo,
                    record_id=rid,
                    source_kind="custom_bound_function_tool",
                    api="callable",
                    file=fn_info.file,
                    line=fn_info.line,
                    tool_name=fn_info.name,
                    description=fn_info.docstring,
                    signature=fn_info.signature,
                    parameters="",
                    input_schema="",
                    output_schema="",
                    return_annotation=fn_info.return_annotation,
                    bound_agent=agent_name,
                    binding_target=f"agent_tools:{agent_name}",
                    extraction_rule="function_bound_in_agent_tools",
                    confidence="medium",
                )
        if isinstance(node, ast.Call):
            return self.extract_tool_from_call(node, bound_agent=agent_name)
        if isinstance(node, ast.Dict):
            return self.tool_from_schema_dict(node, bound_agent=agent_name, api="dict_schema")
        return None

    def extract_openai_tool_schemas(self, node: ast.Call, bound_agent: str) -> None:
        for kw in node.keywords:
            if kw.arg not in {"tools", "functions"}:
                continue
            for item in self.collection_elts(kw.value) or []:
                record = self.tool_from_schema_dict(item, bound_agent=bound_agent, api=f"openai_{kw.arg}")
                if record:
                    self.add_tool(record)

    def tool_from_schema_dict(self, node: ast.AST, bound_agent: str, api: str) -> ToolRecord | None:
        mapping = static_dict(node, self.info, self.index)
        if not mapping:
            return None
        function = mapping.get("function") if isinstance(mapping.get("function"), dict) else mapping
        if not isinstance(function, dict):
            return None
        name = sequence_to_text(function.get("name"))
        description = sequence_to_text(function.get("description"))
        parameters = json.dumps(function.get("parameters", ""), ensure_ascii=False) if function.get("parameters") else ""
        if not (name or description or parameters):
            return None
        rid = add_hash_id("tool", f"custom:{self.repo}:{self.info.path}:{name}:{description}:{bound_agent}")
        return ToolRecord(
            dataset_framework=self.dataset_framework,
            detected_framework="custom",
            repo=self.repo,
            record_id=rid,
            source_kind="custom_openai_tool_schema",
            api=api,
            file=rel(self.info.path, self.info.repo_dir),
            line=getattr(node, "lineno", 0),
            tool_name=name,
            description=description,
            signature="",
            parameters=parameters,
            input_schema=parameters,
            output_schema="",
            return_annotation="",
            bound_agent=bound_agent,
            binding_target=f"{api}:{bound_agent}",
            extraction_rule="openai_tool_schema_bound_to_llm_call",
            confidence="medium",
        )


def file_has_extraction_signal(path: Path, source: str) -> bool:
    if EXTRACTION_SIGNAL_RE.search(source):
        return True
    if SIGNAL_FILE_NAME_RE.search(path.stem) and re.search(r"(agent|tool|prompt|system|instructions|llm)", source, re.I):
        return True
    return False


def resolve_module_path(module_paths: dict[str, Path], module_name: str) -> Path | None:
    if module_name in module_paths:
        return module_paths[module_name]
    matches = [path for module, path in module_paths.items() if module.endswith(f".{module_name}")]
    if len(matches) == 1:
        return matches[0]
    basename_matches = [path for module, path in module_paths.items() if module.split(".")[-1] == module_name]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def parse_repo(dataset_framework: str, repo: str, repo_dir: Path, exclude_non_impl_paths: bool) -> dict[str, Any]:
    index = RepoIndex(repo_dir)
    syntax_errors = 0
    read_errors = 0
    py_files = 0
    skipped_bundled = 0
    skipped_large = 0
    sources: dict[Path, str] = {}
    module_paths: dict[str, Path] = {}
    selected_paths: set[Path] = set()

    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        if is_bundled_framework_source(path, repo_dir):
            skipped_bundled += 1
            continue
        try:
            if path.stat().st_size > MAX_SOURCE_BYTES:
                skipped_large += 1
                continue
        except OSError:
            pass
        py_files += 1
        source = read_text(path)
        if source is None:
            read_errors += 1
            continue
        sources[path] = source
        module_paths[module_name_for(path, repo_dir)] = path
        if file_has_extraction_signal(path, source):
            selected_paths.add(path)

    parsed_paths: set[Path] = set()
    attempted_paths: set[Path] = set()
    queue = list(sorted(selected_paths, key=lambda p: rel(p, repo_dir)))
    while queue:
        path = queue.pop(0)
        if path in attempted_paths:
            continue
        attempted_paths.add(path)
        source = sources.get(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, ValueError):
            syntax_errors += 1
            continue
        info = make_empty_module_info(repo_dir, path, source)
        index.add_module(info)
        IndexBuilder(info, index).visit(tree)
        parsed_paths.add(path)
        for module_name, _imported_name in info.local_imports.values():
            target_path = resolve_module_path(module_paths, module_name)
            target_source = sources.get(target_path) if target_path else None
            if (
                target_path
                and target_path not in attempted_paths
                and target_source
                and (SIGNAL_FILE_NAME_RE.search(target_path.stem) or file_has_extraction_signal(target_path, target_source))
            ):
                queue.append(target_path)

    for info in index.modules.values():
        try:
            tree = ast.parse(info.source, filename=str(info.path))
        except (SyntaxError, ValueError):
            continue
        ToolPreIndexer(dataset_framework, repo, info, index).visit(tree)

    agent_records: list[AgentRecord] = []
    tool_records: list[ToolRecord] = []
    for info in index.modules.values():
        try:
            tree = ast.parse(info.source, filename=str(info.path))
        except (SyntaxError, ValueError):
            continue
        analyzer = RepoAnalyzer(dataset_framework, repo, info, index)
        analyzer.visit(tree)
        analyzer.add_autogpt_file_record()
        agent_records.extend(analyzer.agent_records)
        tool_records.extend(analyzer.tool_records)

    agent_records = dedupe_records(agent_records)
    tool_records = dedupe_records(tool_records)
    agent_records = assign_record_ids(agent_records, "agent")
    tool_records = assign_record_ids(tool_records, "tool")

    return {
        "dataset_framework": dataset_framework,
        "repo": repo,
        "py_files": py_files,
        "skipped_bundled_files": skipped_bundled,
        "skipped_large_files": skipped_large,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "agent_records": agent_records,
        "tool_records": tool_records,
    }


def dedupe_records(records: list[Any]) -> list[Any]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[Any] = []
    for record in records:
        data = asdict(record)
        key = (
            data.get("dataset_framework"),
            data.get("detected_framework"),
            data.get("repo"),
            data.get("source_kind"),
            data.get("api"),
            data.get("file"),
            data.get("line"),
            data.get("content") or data.get("description") or data.get("tool_name"),
            data.get("binding_target"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def assign_record_ids(records: list[Any], prefix: str) -> list[Any]:
    output: list[Any] = []
    for idx, record in enumerate(sorted(records, key=lambda r: (r.file, r.line, r.detected_framework, r.source_kind)), 1):
        output.append(replace(record, record_id=f"{record.dataset_framework}/{record.repo}/{prefix}-{idx:04d}"))
    return output


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


def write_outputs(results: list[dict[str, Any]], out_dir: Path, audit_samples_per_framework: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_path = out_dir / "agent_description_records.jsonl"
    tool_path = out_dir / "tool_description_records.jsonl"
    summary_path = out_dir / "repo_description_summary.csv"
    audit_path = out_dir / "extraction_audit_samples.md"

    all_agents = [record for row in results for record in row["agent_records"]]
    all_tools = [record for row in results for record in row["tool_records"]]

    with agent_path.open("w", encoding="utf-8") as f:
        for record in all_agents:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with tool_path.open("w", encoding="utf-8") as f:
        for record in all_tools:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_framework",
                "repo",
                "py_files",
                "skipped_bundled_files",
                "skipped_large_files",
                "read_errors",
                "syntax_errors",
                "agent_record_count",
                "tool_record_count",
                "detected_frameworks",
                "has_agent_description",
                "has_tool_description",
            ],
        )
        writer.writeheader()
        for row in results:
            detected = sorted(
                {
                    record.detected_framework
                    for record in [*row["agent_records"], *row["tool_records"]]
                    if record.detected_framework
                }
            )
            writer.writerow(
                {
                    "dataset_framework": row["dataset_framework"],
                    "repo": row["repo"],
                    "py_files": row["py_files"],
                    "skipped_bundled_files": row["skipped_bundled_files"],
                    "skipped_large_files": row.get("skipped_large_files", 0),
                    "read_errors": row["read_errors"],
                    "syntax_errors": row["syntax_errors"],
                    "agent_record_count": len(row["agent_records"]),
                    "tool_record_count": len(row["tool_records"]),
                    "detected_frameworks": ";".join(detected),
                    "has_agent_description": bool(row["agent_records"]),
                    "has_tool_description": bool(row["tool_records"]),
                }
            )

    write_audit_samples(all_agents, all_tools, audit_path, audit_samples_per_framework)


def write_audit_samples(agents: list[AgentRecord], tools: list[ToolRecord], path: Path, per_framework: int) -> None:
    rng = random.Random(20260621)
    grouped: dict[str, list[Any]] = {}
    for record in [*agents, *tools]:
        grouped.setdefault(record.detected_framework, []).append(record)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Precise Agent/Tool Description Extraction Audit Samples\n\n")
        f.write(f"Sampling: up to {per_framework} mixed agent/tool records per detected framework.\n\n")
        for framework in sorted(grouped):
            records = grouped[framework]
            sample = rng.sample(records, min(per_framework, len(records)))
            f.write(f"## {FRAMEWORK_DISPLAY.get(framework, framework)}\n\n")
            for record in sample:
                data = asdict(record)
                kind = "Agent" if isinstance(record, AgentRecord) else "Tool"
                f.write(f"### {kind}: `{data.get('repo')}` / `{data.get('record_id')}`\n\n")
                f.write(f"- dataset_framework: `{data.get('dataset_framework')}`\n")
                f.write(f"- detected_framework: `{data.get('detected_framework')}`\n")
                f.write(f"- source_kind: `{data.get('source_kind')}`\n")
                f.write(f"- api: `{data.get('api')}`\n")
                f.write(f"- file: `{data.get('file')}`\n")
                f.write(f"- line: `{data.get('line')}`\n")
                f.write(f"- extraction_rule: `{data.get('extraction_rule')}`\n")
                f.write(f"- binding_target: `{data.get('binding_target')}`\n\n")
                text = data.get("content") or data.get("description") or data.get("signature") or ""
                f.write("```text\n")
                f.write(str(text)[:1600].replace("```", "` ` `"))
                f.write("\n```\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/agent_tool_descriptions_precise_v1")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument("--exclude-non-impl-paths", action="store_true")
    parser.add_argument("--audit-samples-per-framework", type=int, default=5)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = Path(args.out_dir).resolve()
    allowed_root = Path("/home/zhaiyibo/zyb").resolve()
    if str(base).startswith("/home/") or str(out_dir).startswith("/home/"):
        if not str(base).startswith(str(allowed_root)) or not str(out_dir).startswith(str(allowed_root)):
            raise SystemExit("Refusing to read/write outside /home/zhaiyibo/zyb")

    frameworks = args.framework or DEFAULT_FRAMEWORKS
    repos = list_repos(base, frameworks, limit=args.limit)
    print(f"Analyzing {len(repos)} repositories: {', '.join(frameworks)}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(parse_repo, dataset_framework, repo, Path(repo_path), args.exclude_non_impl_paths)
            for dataset_framework, repo, repo_path in repos
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if idx % 100 == 0:
                print(f"Completed {idx}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["dataset_framework"], row["repo"]))
    write_outputs(results, out_dir, args.audit_samples_per_framework)
    print(f"Wrote precise extraction results to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
