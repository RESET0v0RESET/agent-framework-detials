#!/usr/bin/env python3
"""Supplement BP2 raw extraction with prompt assets and prompt-loader evidence.

This script does not classify BP2 defects. It extracts additional natural
language evidence needed by LLM-assisted analysis: prompt files, structured
configuration prompts, Python prompt constants, and static prompt-file reads.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    tomllib = None

from rq3_filters import NON_IMPL_DIR_PARTS, SKIP_DIRS, is_excluded_path, is_mock_like_name, iter_py_files


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

AGENT_FIELD_NAMES = {
    "agent_name",
    "agent_role",
    "agent_goal",
    "agent_backstory",
    "ai_name",
    "ai_role",
    "ai_goals",
    "assistant_role_name",
    "user_role_name",
    "critic_role_name",
    "name",
    "role",
    "goal",
    "backstory",
    "background",
    "profile",
    "persona",
    "description",
    "desc",
    "constraints",
    "constraint",
    "critic_criteria",
    "context",
    "system_prompt",
    "system_message",
    "instructions",
    "instruction",
    "prompt",
    "task_prompt",
    "user_prompt",
    "state_prompt",
    "handoff_description",
    "state_modifier",
    "messages_modifier",
}

POSITIONAL_FIELDS = {
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
    "constraints",
    "constraint",
    "critic_criteria",
    "context",
    "assistant_role_name",
    "user_role_name",
    "critic_role_name",
    "task_prompt",
    "system_prompt",
    "system_message",
    "instructions",
    "instruction",
    "prompt",
    "user_prompt",
    "state_prompt",
    "state_modifier",
    "messages_modifier",
]

PROMPT_EXTENSIONS = {
    ".txt",
    ".md",
    ".mdx",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".prompt",
    ".jinja",
    ".jinja2",
    ".j2",
    ".tmpl",
    ".tpl",
}

STRUCTURED_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}

PROMPT_PATH_RE = re.compile(
    r"(^|[_\-. /\\])("
    r"prompt|prompts|systemprompt|system_prompt|system-message|system_message|instruction|instructions|"
    r"persona|personas|role|roles|agent|agents|assistant|assistants"
    r")($|[_\-. /\\])",
    re.IGNORECASE,
)

PROMPT_NAME_RE = re.compile(
    r"("
    r"prompt|system_prompt|systemprompt|system_message|systemmessage|instruction|instructions|"
    r"persona|role|goal|backstory|constraint|context|agent_description|agent_desc"
    r")",
    re.IGNORECASE,
)

HIGH_CONF_PROMPT_NAME_RE = re.compile(
    r"("
    r"system_?prompt|system_?message|developer_?prompt|agent_?prompt|assistant_?prompt|"
    r"agent_?instructions?|assistant_?instructions?|role|goal|backstory|profile|constraints?|"
    r"agent_?description|agent_?desc"
    r")",
    re.IGNORECASE,
)

PROMPT_KEY_RE = re.compile(
    r"^(system_prompt|systemPrompt|system_message|systemMessage|instructions?|instruction|prompt|"
    r"role|goal|backstory|profile|description|constraints?|context|persona)$"
)

LOADER_NAME_RE = re.compile(r"(load_?prompt|prompt_?loader|get_?prompt|read_?prompt)", re.IGNORECASE)
PROMPT_BUILDER_NAMES = {
    "from_messages",
    "from_template",
    "PromptTemplate",
    "ChatPromptTemplate",
    "SystemMessage",
    "SystemMessagePromptTemplate",
    "HumanMessagePromptTemplate",
}
AGENT_DESCRIPTION_TEXT_SIGNAL_RE = re.compile(
    r"\b("
    r"you are|your role|your goal|as an agent|as a .*assistant|assistant|agent|"
    r"persona|role|goal|backstory|constraints?"
    r")\b",
    re.IGNORECASE,
)

MAX_FILE_BYTES = 512_000
MAX_CONTENT_CHARS = 12_000
MIN_PROMPT_CHARS = 20


@dataclass(frozen=True)
class PromptAssetRecord:
    framework: str
    repo: str
    source_kind: str
    file: str
    line: int
    key: str
    ref_path: str
    content: str
    content_sha1: str
    char_count: int
    truncated: bool
    code: str


def read_text(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None


def rel(path: Path, repo_dir: Path) -> str:
    try:
        return path.relative_to(repo_dir).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_content(text: str) -> tuple[str, str, int, bool]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    char_count = len(text)
    truncated = char_count > MAX_CONTENT_CHARS
    content = text[:MAX_CONTENT_CHARS] if truncated else text
    sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return content, sha1, char_count, truncated


def make_record(
    framework: str,
    repo: str,
    repo_dir: Path,
    source_kind: str,
    file: Path,
    line: int,
    key: str,
    ref_path: str,
    content_text: str,
    code: str = "",
) -> PromptAssetRecord | None:
    if len(content_text.strip()) < MIN_PROMPT_CHARS:
        return None
    content, sha1, char_count, truncated = normalize_content(content_text.strip())
    return PromptAssetRecord(
        framework=framework,
        repo=repo,
        source_kind=source_kind,
        file=rel(file, repo_dir),
        line=line,
        key=key,
        ref_path=ref_path,
        content=content,
        content_sha1=sha1,
        char_count=char_count,
        truncated=truncated,
        code=code.strip()[:400],
    )


def is_high_conf_prompt_text(name_or_key: str, text: str) -> bool:
    if HIGH_CONF_PROMPT_NAME_RE.search(name_or_key):
        return True
    return bool(AGENT_DESCRIPTION_TEXT_SIGNAL_RE.search(text))


def make_prompt_record(
    framework: str,
    repo: str,
    repo_dir: Path,
    source_kind: str,
    file: Path,
    line: int,
    key: str,
    ref_path: str,
    content_text: str,
    code: str = "",
) -> PromptAssetRecord | None:
    if not is_high_conf_prompt_text(key, content_text):
        return None
    return make_record(framework, repo, repo_dir, source_kind, file, line, key, ref_path, content_text, code)


def should_skip_dir(dirname: str, exclude_non_impl_paths: bool) -> bool:
    lower = dirname.lower()
    if dirname in SKIP_DIRS or dirname.startswith("."):
        return True
    if exclude_non_impl_paths and lower in NON_IMPL_DIR_PARTS:
        return True
    return False


def is_prompt_asset_path(path: Path, repo_dir: Path) -> bool:
    if path.suffix.lower() not in PROMPT_EXTENSIONS:
        return False
    rel_path = rel(path, repo_dir)
    return bool(PROMPT_PATH_RE.search(rel_path))


def iter_candidate_asset_files(repo_dir: Path, exclude_non_impl_paths: bool) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if not should_skip_dir(d, exclude_non_impl_paths)]
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() not in PROMPT_EXTENSIONS:
                continue
            if exclude_non_impl_paths:
                try:
                    if is_excluded_path(path, repo_dir) and not is_prompt_asset_path(path, repo_dir):
                        continue
                except Exception:
                    continue
            yield path


def strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    return value


def extract_yaml_like_key_values(text: str) -> list[tuple[str, str, int]]:
    records: list[tuple[str, str, int]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
        if not match or not PROMPT_KEY_RE.match(match.group(1)):
            i += 1
            continue
        key = match.group(1)
        rest = match.group(2).strip()
        start_line = i + 1
        if rest in {"|", ">"}:
            base_indent = len(line) - len(line.lstrip())
            chunks: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() and len(next_line) - len(next_line.lstrip()) <= base_indent:
                    break
                chunks.append(next_line.strip("\n"))
                i += 1
            records.append((key, "\n".join(chunks).strip(), start_line))
            continue
        records.append((key, strip_yaml_scalar(rest), start_line))
        i += 1
    return records


def flatten_structured(obj: object, prefix: str = "") -> Iterable[tuple[str, object]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}" if prefix else key_str
            yield path, value
            yield from flatten_structured(value, path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from flatten_structured(value, f"{prefix}[{idx}]")


def extract_structured_prompts(path: Path, text: str) -> list[tuple[str, str, int]]:
    suffix = path.suffix.lower()
    extracted: list[tuple[str, str, int]] = []
    if suffix == ".json":
        try:
            obj = json.loads(text)
            for key_path, value in flatten_structured(obj):
                leaf = re.split(r"[.\[]", key_path)[-1].rstrip("]")
                if PROMPT_KEY_RE.match(leaf) and isinstance(value, str):
                    extracted.append((key_path, value, 0))
            return extracted
        except Exception:
            pass
    if suffix == ".toml" and tomllib is not None:
        try:
            obj = tomllib.loads(text)
            for key_path, value in flatten_structured(obj):
                leaf = re.split(r"[.\[]", key_path)[-1].rstrip("]")
                if PROMPT_KEY_RE.match(leaf) and isinstance(value, str):
                    extracted.append((key_path, value, 0))
            return extracted
        except Exception:
            pass
    if suffix in {".yaml", ".yml", ".toml"}:
        return extract_yaml_like_key_values(text)
    return extracted


def extract_asset_file_records(framework: str, repo: str, repo_dir: Path, path: Path) -> list[PromptAssetRecord]:
    text = read_text(path)
    if text is None:
        return []
    records: list[PromptAssetRecord] = []
    structured = extract_structured_prompts(path, text) if path.suffix.lower() in STRUCTURED_EXTENSIONS else []
    for key, value, line in structured:
        record = make_prompt_record(framework, repo, repo_dir, "structured_prompt_key", path, line, key, "", value)
        if record:
            records.append(record)
    if records:
        return records
    if is_prompt_asset_path(path, repo_dir):
        record = make_record(framework, repo, repo_dir, "prompt_asset_file", path, 0, path.stem, rel(path, repo_dir), text)
        if record:
            records.append(record)
    return records


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


def static_string(node: ast.AST | None, names: dict[str, str] | None = None) -> str | None:
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
        left = static_string(node.left, names)
        right = static_string(node.right, names)
        if left is not None and right is not None:
            return left + right
    if names and isinstance(node, ast.Name):
        return names.get(node.id)
    return None


def static_list_of_strings(node: ast.AST | None, names: dict[str, str] | None = None) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for elt in node.elts:
            value = static_string(elt, names)
            if value is None:
                return None
            values.append(value)
        return values
    value = static_string(node, names)
    if value is not None:
        return [value]
    return None


def static_field_value(node: ast.AST | None, names: dict[str, str] | None = None) -> str | None:
    value = static_string(node, names)
    if value is not None:
        return value
    values = static_list_of_strings(node, names)
    if values:
        return "\n".join(f"- {value}" for value in values)
    return None


def static_dict(node: ast.AST | None, names: dict[str, str] | None = None) -> dict[str, str] | None:
    if not isinstance(node, ast.Dict):
        return None
    result: dict[str, str] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = static_string(key_node, names)
        if key is None:
            continue
        value = static_field_value(value_node, names)
        if value is not None:
            result[key] = value
    return result


def static_nested_dict(node: ast.AST | None, names: dict[str, str] | None = None) -> dict[str, object] | None:
    if not isinstance(node, ast.Dict):
        return None
    result: dict[str, object] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = static_string(key_node, names)
        if key is None:
            continue
        nested = static_nested_dict(value_node, names)
        if nested is not None:
            result[key] = nested
            continue
        value = static_field_value(value_node, names)
        if value is not None:
            result[key] = value
    return result


def first_static_string(node: ast.AST | None, names: dict[str, str] | None = None) -> str | None:
    value = static_string(node, names)
    if value is not None:
        return value
    if isinstance(node, ast.Call):
        for arg in node.args:
            value = first_static_string(arg, names)
            if value is not None:
                return value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            value = first_static_string(elt, names)
            if value is not None:
                return value
    return None


def unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)[:500]
    except Exception:
        return ast.dump(node)[:500]


def collect_static_strings(node: ast.AST | None, names: dict[str, str], chunks: list[str]) -> None:
    if node is None:
        return
    value = static_string(node, names)
    if value is not None:
        chunks.append(value)
        return
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            collect_static_strings(elt, names, chunks)
        return
    if isinstance(node, ast.Dict):
        for key_node, value_node in zip(node.keys, node.values):
            collect_static_strings(key_node, names, chunks)
            collect_static_strings(value_node, names, chunks)
        return
    if isinstance(node, ast.Call):
        full, last = call_name(node.func)
        if last in PROMPT_BUILDER_NAMES or (full and any(part in PROMPT_BUILDER_NAMES for part in full.split("."))):
            for arg in node.args:
                collect_static_strings(arg, names, chunks)
            for kw in node.keywords:
                collect_static_strings(kw.value, names, chunks)


def extract_prompt_builder_texts(node: ast.Call, names: dict[str, str]) -> list[tuple[str, str]]:
    full, last = call_name(node.func)
    if not last:
        return []

    records: list[tuple[str, str]] = []

    if last in {"SystemMessage", "SystemMessagePromptTemplate"}:
        chunks: list[str] = []
        for arg in node.args:
            collect_static_strings(arg, names, chunks)
        for kw in node.keywords:
            if kw.arg in {None, "content", "prompt", "template"}:
                collect_static_strings(kw.value, names, chunks)
        text = "\n".join(chunk for chunk in chunks if chunk.strip())
        if text:
            records.append(("system_message", text))
        return records

    if last == "from_template" or last == "PromptTemplate":
        chunks = []
        for arg in node.args:
            collect_static_strings(arg, names, chunks)
        for kw in node.keywords:
            if kw.arg in {None, "template", "prompt"}:
                collect_static_strings(kw.value, names, chunks)
        text = "\n".join(chunk for chunk in chunks if chunk.strip())
        if text:
            records.append(("prompt_template", text))
        return records

    if last == "from_messages":
        for arg in node.args:
            records.extend(extract_message_list_texts(arg, names))
        for kw in node.keywords:
            records.extend(extract_message_list_texts(kw.value, names))
        return records

    if last == "ChatPromptTemplate" or (full and full.endswith("ChatPromptTemplate")):
        chunks = []
        for arg in node.args:
            collect_static_strings(arg, names, chunks)
        for kw in node.keywords:
            collect_static_strings(kw.value, names, chunks)
        text = "\n".join(chunk for chunk in chunks if chunk.strip())
        if text:
            records.append(("chat_prompt_template", text))
        return records

    return records


def extract_message_list_texts(node: ast.AST, names: dict[str, str]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        elements = node.elts
    else:
        elements = [node]

    for element in elements:
        if isinstance(element, (ast.Tuple, ast.List)) and len(element.elts) >= 2:
            role = static_string(element.elts[0], names)
            content = static_string(element.elts[1], names)
            if role and content:
                role_lower = role.lower()
                if role_lower in {"system", "developer"}:
                    records.append((f"{role_lower}_message", content))
                elif role_lower in {"assistant", "ai"} and is_high_conf_prompt_text(role_lower, content):
                    records.append((f"{role_lower}_message", content))
            continue
        if isinstance(element, ast.Call):
            records.extend(extract_prompt_builder_texts(element, names))
    return records


def line_text(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()[:400]
    return ""


def resolve_repo_path(raw_path: str, repo_dir: Path, current_file: Path) -> Path | None:
    if not raw_path or "://" in raw_path:
        return None
    raw_path = raw_path.strip()
    if not raw_path or any(ch in raw_path for ch in "\n\r{}"):
        return None
    candidates = []
    p = Path(raw_path)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((current_file.parent / p).resolve())
        candidates.append((repo_dir / p).resolve())
        for parent in current_file.parents:
            try:
                if not str(parent.resolve()).startswith(str(repo_dir.resolve())):
                    continue
            except Exception:
                continue
            candidates.append((parent / p).resolve())
    repo_resolved = repo_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if str(resolved).startswith(str(repo_resolved)) and resolved.is_file():
                return resolved
        except Exception:
            continue
    basename = p.name
    if basename and len(basename) > 3:
        try:
            for candidate in repo_dir.rglob(basename):
                resolved = candidate.resolve()
                if not str(resolved).startswith(str(repo_resolved)):
                    continue
                if resolved.is_file() and raw_path.replace("\\", "/").split("/")[-1] == resolved.name:
                    return resolved
        except Exception:
            return None
    return None


def call_loads_prompt_asset(node: ast.Call, names: dict[str, str], repo_dir: Path, current_file: Path) -> tuple[Path | None, str]:
    full, last = call_name(node.func)
    full_l = full or ""
    last_l = last or ""

    # Path("...").read_text() and pathlib.Path("...").read_text().
    if last_l == "read_text" and isinstance(node.func, ast.Attribute):
        target = node.func.value
        if isinstance(target, ast.Call):
            target_full, target_last = call_name(target.func)
            if target_last == "Path" or target_full in {"pathlib.Path", "Path"}:
                raw = first_static_string(target, names)
                if raw:
                    return resolve_repo_path(raw, repo_dir, current_file), raw

    # open("...").read().
    if last_l == "read" and isinstance(node.func, ast.Attribute):
        inner = node.func.value
        if isinstance(inner, ast.Call):
            _, inner_last = call_name(inner.func)
            if inner_last == "open":
                raw = first_static_string(inner, names)
                if raw:
                    return resolve_repo_path(raw, repo_dir, current_file), raw

    # load_prompt("..."), get_prompt("..."), prompt_loader.load("...").
    if LOADER_NAME_RE.search(full_l) or LOADER_NAME_RE.search(last_l):
        raw = first_static_string(node, names)
        if raw:
            return resolve_repo_path(raw, repo_dir, current_file), raw

    # json.load(open("...")), yaml.safe_load(open("...")).
    if last_l in {"load", "safe_load"} and node.args:
        raw = first_static_string(node.args[0], names)
        if raw:
            return resolve_repo_path(raw, repo_dir, current_file), raw

    return None, ""


class PythonPromptAnalyzer(ast.NodeVisitor):
    def __init__(self, framework: str, repo: str, repo_dir: Path, path: Path, source: str):
        self.framework = framework
        self.repo = repo
        self.repo_dir = repo_dir
        self.path = path
        self.lines = source.splitlines()
        self.roots = OFFICIAL_ROOTS_BY_FRAMEWORK[framework]
        self.alias_to_qualified: dict[str, str] = {}
        self.official_module_aliases: set[str] = set()
        self.names: dict[str, str] = {}
        self.dict_vars: dict[str, dict[str, object]] = {}
        self.file_agent_fields: dict[str, str] = {}
        self.records: list[PromptAssetRecord] = []

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
        if self.framework == "metagpt" and self.is_metagpt_role_class(node):
            self.add_metagpt_role_class_record(node)
        self.generic_visit(node)

    def is_metagpt_role_class(self, node: ast.ClassDef) -> bool:
        for base in node.bases:
            full, last = call_name(base)
            if last == "Role" and self.is_confirmed_official_call(full, last):
                return True
        return False

    def record_assignment(self, name: str, value_node: ast.AST | None, lineno: int) -> None:
        nested = static_nested_dict(value_node, self.names)
        if nested is not None:
            self.dict_vars[name] = nested

        field_value = static_field_value(value_node, self.names)
        if self.framework == "autogpt" and name in {"ai_name", "ai_role", "ai_goals"} and field_value is not None:
            self.file_agent_fields[name] = field_value

        literal = static_string(value_node, self.names)
        if literal is not None:
            self.names[name] = literal
            if PROMPT_NAME_RE.search(name) and is_high_conf_prompt_text(name, literal):
                record = make_prompt_record(
                    self.framework,
                    self.repo,
                    self.repo_dir,
                    "python_prompt_constant",
                    self.path,
                    lineno,
                    name,
                    "",
                    literal,
                    line_text(self.lines, lineno),
                )
                if record:
                    self.records.append(record)
            return

        if field_value is not None and name in AGENT_FIELD_NAMES and PROMPT_NAME_RE.search(name):
            record = make_prompt_record(
                self.framework,
                self.repo,
                self.repo_dir,
                "python_prompt_constant",
                self.path,
                lineno,
                name,
                "",
                field_value,
                line_text(self.lines, lineno),
            )
            if record:
                self.records.append(record)
            return

        if isinstance(value_node, ast.Call) and PROMPT_NAME_RE.search(name):
            asset_path, raw_ref = call_loads_prompt_asset(value_node, self.names, self.repo_dir, self.path)
            if asset_path:
                content = read_text(asset_path) if asset_path else None
                if not content:
                    return
                record = make_prompt_record(
                    self.framework,
                    self.repo,
                    self.repo_dir,
                    "python_prompt_loader_reference",
                    self.path,
                    lineno,
                    name,
                    rel(asset_path, self.repo_dir),
                    content,
                    line_text(self.lines, lineno),
                )
                if record:
                    self.records.append(record)

    def dict_key_name(self, node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def resolve_config_dict(self, node: ast.AST | None) -> dict[str, str]:
        if node is None:
            return {}
        flat = static_dict(node, self.names)
        if flat:
            return {key: value for key, value in flat.items() if key in AGENT_FIELD_NAMES}
        if isinstance(node, ast.Name):
            obj = self.dict_vars.get(node.id)
            if isinstance(obj, dict):
                return {key: value for key, value in obj.items() if key in AGENT_FIELD_NAMES and isinstance(value, str)}
        if isinstance(node, ast.Subscript):
            root_name = self.dict_key_name(node.value)
            index = static_string(node.slice, self.names)
            if root_name and index:
                root = self.dict_vars.get(root_name)
                if isinstance(root, dict):
                    obj = root.get(index)
                    if isinstance(obj, dict):
                        return {
                            key: value
                            for key, value in obj.items()
                            if key in AGENT_FIELD_NAMES and isinstance(value, str)
                        }
        return {}

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.record_assignment(target.id, node.value, getattr(node, "lineno", 0))
            elif isinstance(target, ast.Attribute):
                attr = target.attr
                if PROMPT_NAME_RE.search(attr) or isinstance(node.value, ast.Dict):
                    self.record_assignment(attr, node.value, getattr(node, "lineno", 0))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.record_assignment(node.target.id, node.value, getattr(node, "lineno", 0))
        elif isinstance(node.target, ast.Attribute):
            if PROMPT_NAME_RE.search(node.target.attr) or isinstance(node.value, ast.Dict):
                self.record_assignment(node.target.attr, node.value, getattr(node, "lineno", 0))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        full, last = call_name(node.func)
        if self.is_confirmed_official_call(full, last) and last in AGENT_APIS.get(self.framework, set()):
            self.add_agent_description_record(last or "", node)

        for key, content in extract_prompt_builder_texts(node, self.names):
            record = make_prompt_record(
                self.framework,
                self.repo,
                self.repo_dir,
                "python_prompt_builder",
                self.path,
                getattr(node, "lineno", 0),
                key,
                "",
                content,
                line_text(self.lines, getattr(node, "lineno", 0)),
            )
            if record:
                self.records.append(record)

        asset_path, raw_ref = call_loads_prompt_asset(node, self.names, self.repo_dir, self.path)
        if raw_ref and asset_path:
            content = read_text(asset_path) if asset_path else None
            if not content:
                self.generic_visit(node)
                return
            record = make_prompt_record(
                self.framework,
                self.repo,
                self.repo_dir,
                "python_prompt_file_read",
                self.path,
                getattr(node, "lineno", 0),
                "file_read",
                rel(asset_path, self.repo_dir),
                content,
                line_text(self.lines, getattr(node, "lineno", 0)),
            )
            if record:
                self.records.append(record)
        self.generic_visit(node)

    def add_agent_description_record(self, api: str, node: ast.Call) -> None:
        fields: dict[str, str] = {}
        exprs: dict[str, str] = {}

        for pos, field in POSITIONAL_FIELDS.get(api, {}).items():
            if pos < len(node.args):
                value = static_field_value(node.args[pos], self.names)
                if value is not None:
                    fields[field] = value
                exprs[field] = unparse(node.args[pos])

        for kw in node.keywords:
            if kw.arg == "config":
                fields.update(self.resolve_config_dict(kw.value))
                exprs["config"] = unparse(kw.value)
                continue
            if kw.arg not in AGENT_FIELD_NAMES:
                continue
            value = static_field_value(kw.value, self.names)
            if value is not None:
                fields[kw.arg] = value
            exprs[kw.arg] = unparse(kw.value)

        if not fields:
            return

        chunks = []
        for key in AGENT_FIELD_PRIORITY:
            if key in fields:
                chunks.append(f"{key}: {fields[key]}")
        for key in sorted(fields):
            if key not in AGENT_FIELD_PRIORITY:
                chunks.append(f"{key}: {fields[key]}")
        content = "\n".join(chunks)

        record = make_record(
            self.framework,
            self.repo,
            self.repo_dir,
            "framework_agent_description",
            self.path,
            getattr(node, "lineno", 0),
            api,
            "",
            content,
            line_text(self.lines, getattr(node, "lineno", 0)),
        )
        if record:
            self.records.append(record)

    def add_file_agent_field_record(self) -> None:
        if self.framework != "autogpt" or not self.file_agent_fields:
            return
        chunks = []
        for key in AGENT_FIELD_PRIORITY:
            if key in self.file_agent_fields:
                chunks.append(f"{key}: {self.file_agent_fields[key]}")
        record = make_record(
            self.framework,
            self.repo,
            self.repo_dir,
            "framework_agent_description",
            self.path,
            0,
            "AutoGPTAISpec",
            "",
            "\n".join(chunks),
            "",
        )
        if record:
            self.records.append(record)

    def add_metagpt_role_class_record(self, node: ast.ClassDef) -> None:
        fields: dict[str, str] = {"name": node.name}
        local_names = dict(self.names)

        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            args = list(item.args.posonlyargs) + list(item.args.args)
            if args and args[0].arg == "self":
                args = args[1:]
            defaults = list(item.args.defaults)
            if defaults:
                for arg, default_node in zip(args[-len(defaults) :], defaults):
                    value = static_field_value(default_node, local_names)
                    if value is not None:
                        local_names[arg.arg] = value
                        if arg.arg in AGENT_FIELD_NAMES:
                            fields[arg.arg] = value
            for arg, default_node in zip(item.args.kwonlyargs, item.args.kw_defaults):
                value = static_field_value(default_node, local_names)
                if value is not None:
                    local_names[arg.arg] = value
                    if arg.arg in AGENT_FIELD_NAMES:
                        fields[arg.arg] = value

            for child in ast.walk(item):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Attribute) and target.attr in AGENT_FIELD_NAMES:
                            value = static_field_value(child.value, local_names)
                            if value is not None:
                                fields[target.attr] = value
                if not isinstance(child, ast.Call):
                    continue
                _, last = call_name(child.func)
                is_super_init = (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr == "__init__"
                    and isinstance(child.func.value, ast.Call)
                    and call_name(child.func.value.func)[1] == "super"
                )
                if last != "__init__" or not is_super_init:
                    continue
                for pos, field in POSITIONAL_FIELDS.get("Role", {}).items():
                    if pos < len(child.args):
                        value = static_field_value(child.args[pos], local_names)
                        if value is not None:
                            fields[field] = value
                for kw in child.keywords:
                    if kw.arg in AGENT_FIELD_NAMES:
                        value = static_field_value(kw.value, local_names)
                        if value is not None:
                            fields[kw.arg] = value

        if len(fields) <= 1:
            return

        chunks = []
        for key in AGENT_FIELD_PRIORITY:
            if key in fields:
                chunks.append(f"{key}: {fields[key]}")
        for key in sorted(fields):
            if key not in AGENT_FIELD_PRIORITY:
                chunks.append(f"{key}: {fields[key]}")

        record = make_record(
            self.framework,
            self.repo,
            self.repo_dir,
            "framework_agent_description",
            self.path,
            getattr(node, "lineno", 0),
            "RoleClass",
            "",
            "\n".join(chunks),
            line_text(self.lines, getattr(node, "lineno", 0)),
        )
        if record:
            self.records.append(record)


def extract_python_records(framework: str, repo: str, repo_dir: Path, path: Path) -> tuple[list[PromptAssetRecord], str | None]:
    source = read_text(path)
    if source is None:
        return [], "read_error"
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [], "syntax_error"
    analyzer = PythonPromptAnalyzer(framework, repo, repo_dir, path, source)
    analyzer.visit(tree)
    analyzer.add_file_agent_field_record()
    return analyzer.records, None


def dedupe_records(records: Iterable[PromptAssetRecord]) -> list[PromptAssetRecord]:
    seen: set[tuple[str, str, str, int, str, str, str]] = set()
    unique: list[PromptAssetRecord] = []
    for record in records:
        key = (
            record.source_kind,
            record.file,
            record.key,
            record.line,
            record.ref_path,
            record.content_sha1,
            record.code,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def analyze_repo(args: tuple[str, str, str, bool]) -> dict:
    framework, repo_name, repo_path_str, exclude_non_impl_paths, python_only = args
    repo_dir = Path(repo_path_str)
    records: list[PromptAssetRecord] = []
    py_files = 0
    asset_files = 0
    read_errors = 0
    syntax_errors = 0

    if not python_only:
        for path in iter_candidate_asset_files(repo_dir, exclude_non_impl_paths):
            asset_files += 1
            try:
                records.extend(extract_asset_file_records(framework, repo_name, repo_dir, path))
            except Exception:
                continue

    for path in iter_py_files(repo_dir, exclude_non_impl_paths=exclude_non_impl_paths):
        py_files += 1
        try:
            py_records, error = extract_python_records(framework, repo_name, repo_dir, path)
        except Exception:
            continue
        records.extend(py_records)
        if error == "read_error":
            read_errors += 1
        elif error == "syntax_error":
            syntax_errors += 1

    records = dedupe_records(records)
    return {
        "framework": framework,
        "repo": repo_name,
        "py_files": py_files,
        "asset_files": asset_files,
        "read_errors": read_errors,
        "syntax_errors": syntax_errors,
        "prompt_asset_records": records,
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
    raw_path = out_dir / "bp2_prompt_assets_raw.jsonl"
    repo_csv = out_dir / "bp2_prompt_assets_repo_summary.csv"
    fw_csv = out_dir / "bp2_prompt_assets_summary_by_framework.csv"

    with raw_path.open("w", encoding="utf-8") as f:
        for row in results:
            for record in row["prompt_asset_records"]:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "framework",
                "repo",
                "py_files",
                "asset_files",
                "read_errors",
                "syntax_errors",
                "prompt_asset_record_count",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "framework": row["framework"],
                    "repo": row["repo"],
                    "py_files": row["py_files"],
                    "asset_files": row["asset_files"],
                    "read_errors": row["read_errors"],
                    "syntax_errors": row["syntax_errors"],
                    "prompt_asset_record_count": len(row["prompt_asset_records"]),
                }
            )

    summary_rows = []
    for fw in sorted({row["framework"] for row in results}):
        rows = [row for row in results if row["framework"] == fw]
        records = [record for row in rows for record in row["prompt_asset_records"]]
        by_kind: dict[str, int] = {}
        for record in records:
            by_kind[record.source_kind] = by_kind.get(record.source_kind, 0) + 1
        summary_rows.append(
            {
                "framework": fw,
                "population_repos": len(rows),
                "repos_with_prompt_assets": sum(1 for row in rows if row["prompt_asset_records"]),
                "prompt_asset_record_count": len(records),
                "py_files": sum(int(row["py_files"]) for row in rows),
                "asset_files": sum(int(row["asset_files"]) for row in rows),
                "read_errors": sum(int(row["read_errors"]) for row in rows),
                "syntax_errors": sum(int(row["syntax_errors"]) for row in rows),
                "prompt_asset_file": by_kind.get("prompt_asset_file", 0),
                "structured_prompt_key": by_kind.get("structured_prompt_key", 0),
                "framework_agent_description": by_kind.get("framework_agent_description", 0),
                "python_prompt_builder": by_kind.get("python_prompt_builder", 0),
                "python_prompt_constant": by_kind.get("python_prompt_constant", 0),
                "python_prompt_loader_reference": by_kind.get("python_prompt_loader_reference", 0),
                "python_prompt_file_read": by_kind.get("python_prompt_file_read", 0),
            }
        )
    with fw_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/zhaiyibo/zyb/repo2")
    parser.add_argument("--out-dir", default="/home/zhaiyibo/zyb/rq3_static_repro/results/bp2_prompt_assets_raw")
    parser.add_argument("--framework", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional repo limit per framework for pilot runs")
    parser.add_argument("--python-only", action="store_true", help="Skip non-Python prompt asset files")
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
        futures = [pool.submit(analyze_repo, (*repo, args.exclude_non_impl_paths, args.python_only)) for repo in repos]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 500 == 0:
                print(f"Completed {i}/{len(futures)} repositories", file=sys.stderr)

    results.sort(key=lambda row: (row["framework"], row["repo"]))
    write_outputs(results, out_dir)
    print(f"Wrote prompt asset raw extraction to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
