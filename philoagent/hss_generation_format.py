"""Recover omitted registration metadata from mutually consistent static facts.

This repairs the response envelope only. Candidate source is parsed as AST and
is never imported, rewritten, or executed; creation/validation remain native.
"""

from __future__ import annotations

import ast
import hashlib
import keyword
import re
from typing import Callable


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LABELS = ("Class Name", "Module Name", "Enum Name", "Enum Value")


def _identifier(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(value)) and not keyword.iskeyword(value)


def recover_missing_registration(response_text: str, native_parse: Callable) -> dict | None:
    """Supply missing enum fields only when independent source facts agree.

    Call after the native parser returns ``None``. Complete registration is
    outside this repair's scope; ambiguity raises rather than choosing a name.
    ``native_parse`` must be the original bound native parser, not this hook.
    """
    if not isinstance(response_text, str):
        raise ValueError("Candidate response must be text")
    blocks = list(re.finditer(r"```python\n(.*?)\n```", response_text, re.DOTALL))
    if len(blocks) != 1:
        raise ValueError("Registration recovery requires one complete Python block")
    block = blocks[0]
    outside = response_text[:block.start()] + response_text[block.end():]
    if re.search(r"^\s*```", outside, re.MULTILINE):
        raise ValueError("Registration recovery cannot choose among code blocks")
    fields = {}
    for label in _LABELS:
        matches = re.findall(r"^\s*\*\*" + re.escape(label) + r"\*\*:\s*([^\n]+)$",
                             outside, re.MULTILINE)
        if len(matches) > 1:
            raise ValueError(f"Registration recovery found repeated {label}")
        if matches:
            value = matches[0].strip()
            if not _identifier(value):
                raise ValueError(f"Registration recovery found invalid {label}")
            fields[label] = value
    if "Enum Name" in fields and "Enum Value" in fields:
        return None
    if not all(label in fields for label in ("Class Name", "Module Name")):
        raise ValueError("Registration recovery requires explicit class and module names")

    code = block[1].strip()
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError("Registration recovery requires complete Python AST") from error
    provider_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                        and any(isinstance(base, ast.Name) and base.id == "BaseMemoryProvider"
                                for base in node.bases)]
    if (len(provider_classes) != 1 or provider_classes[0].name != fields["Class Name"]
            or sum(isinstance(node, ast.ClassDef) and node.name == fields["Class Name"]
                   for node in tree.body) != 1):
        raise ValueError("Explicit class does not identify one AST memory provider")

    imports = []
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == "MemoryType":
                    imports.append((node, alias))
        if ((isinstance(node, ast.Name) and node.id == "MemoryType"
             and isinstance(node.ctx, (ast.Store, ast.Del)))
                or (isinstance(node, ast.arg) and node.arg == "MemoryType")
                or (isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "MemoryType")):
            raise ValueError("MemoryType is rebound in candidate source")
        if isinstance(node, ast.Name) and node.id == "MemoryType":
            parent = parents.get(node)
            if (not isinstance(parent, ast.Attribute) or parent.value is not node
                    or not isinstance(parent.ctx, ast.Load)):
                raise ValueError("MemoryType must use explicit unchanged member references")
    if (len(imports) != 1 or not isinstance(imports[0][0], ast.ImportFrom)
            or imports[0][0].module != "EvolveLab.memory_types"
            or imports[0][1].name != "MemoryType" or imports[0][1].asname):
        raise ValueError("Registration recovery requires the native MemoryType import")
    references = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                  and isinstance(node.value, ast.Name) and node.value.id == "MemoryType"}
    if len(references) != 1:
        raise ValueError("Registration recovery requires one distinct MemoryType member")
    enum_name = next(iter(references))

    roots, defaults = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            path = re.fullmatch(r"\./storage/([^/\\]+)(?:/[^\\]*)?", node.value)
            if path:
                if not _identifier(path[1]) or ".." in node.value.split("/"):
                    raise ValueError("Registration recovery found an invalid storage namespace")
                roots.add(path[1])
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "storage_dir"
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            value = node.args[1].value
            if re.fullmatch(r"\./storage/[A-Za-z_][A-Za-z0-9_]*", value):
                defaults.add(value.rsplit("/", 1)[1])
    if len(roots) != 1 or defaults != roots:
        raise ValueError("Registration recovery requires one storage root and matching explicit default")
    enum_value = next(iter(roots))
    if (enum_name != enum_value.upper() or fields["Module Name"] != enum_value + "_provider"
            or fields["Class Name"].casefold() != enum_value.replace("_", "").casefold() + "provider"):
        raise ValueError("Class, module, MemoryType and storage namespace do not agree")
    expected = {"Enum Name": enum_name, "Enum Value": enum_value}
    if any(label in fields and fields[label] != value for label, value in expected.items()):
        raise ValueError("Explicit enum registration conflicts with candidate source")
    added = {label: value for label, value in expected.items() if label not in fields}
    augmented = response_text + "\n" + "\n".join(f"**{key}**: {value}" for key, value in added.items()) + "\n"
    config = native_parse(augmented)
    if (not config or config.get("provider_code") != {"class_name": fields["Class Name"],
            "module_name": fields["Module Name"], "code": code}
            or config.get("memory_type_info") != {"enum_name": enum_name, "enum_value": enum_value}):
        raise ValueError("Native parser did not preserve candidate source and registration")
    # Bold prose inside source is not an explicit configuration declaration.
    declared_settings = re.findall(r"^\s*\*\*([^*]+)\*\*:\s*[^\n]+$", outside, re.MULTILINE)
    if not any(key.strip() not in (*_LABELS, "orientation_items") for key in declared_settings):
        config["config_updates"] = {}
    config["_raw_llm_output"] = response_text
    config["_generation_format_repair"] = {
        "version": 1, "operation": "recover_missing_memory_type_registration",
        "original_response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "provider_code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "added_fields": added, "memory_type_info": {"enum_name": enum_name, "enum_value": enum_value},
        "evidence": {"class_name": fields["Class Name"], "module_name": fields["Module Name"],
                     "memory_type_members": sorted(references), "storage_roots": sorted(roots),
                     "storage_defaults": sorted(defaults)},
        "source_modified": False, "source_executed": False, "configuration_values_inferred": False,
    }
    return config
