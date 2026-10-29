"""Hard factor projections at the native candidate-generation boundary.

The parent mapping and configuration are read as AST literals from the isolated
workspace. No generated program is executed by this policy. Static screening is
a conservative experiment constraint, not an operating-system sandbox.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import functools
import hashlib
import importlib
import json
import keyword
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Callable

from .evidence import append_event
from .orientation import assert_unchanged


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_META_KEYS = {"model", "orientation_items", "philo_orientation_items", "philo_orientation_sha256"}
_SAFE_STDLIB = {
    "__future__", "ast", "collections", "copy", "dataclasses", "datetime", "enum",
    "functools", "hashlib", "heapq", "itertools", "json", "logging", "math", "os",
    "pathlib", "re", "statistics", "string", "time", "typing", "unicodedata", "uuid",
}
_NATIVE_IMPORTS = {"EvolveLab.base_memory", "EvolveLab.memory_types"}
_SAFE_SUBMODULES = {"collections.abc", "os.path", "typing.io", "typing.re"}
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "builtins", "importlib", "subprocess",
    "requests", "urllib", "socket", "httpx", "aiohttp", "openai", "litellm",
}
_FORBIDDEN_CALLS = {
    "system", "popen", "Popen", "spawn", "spawnl", "spawnv", "fork", "execl", "execv",
    "run_command", "check_call", "check_output", "urlopen", "urlretrieve", "connect",
    "OpenAI", "AsyncOpenAI", "OpenAIServerModel", "HTTPConnection", "HTTPSConnection",
    "Client", "Session", "basicConfig", "FileHandler", "TimedRotatingFileHandler",
    "RotatingFileHandler", "getLoggerClass", "setLoggerClass", "dictConfig", "fileConfig",
}
_OS_PATH_METHODS = {
    "join", "exists", "isfile", "isdir", "basename", "dirname", "split", "splitext",
    "normpath", "abspath", "realpath", "relpath", "isabs", "commonpath", "getsize",
}
_LOCAL_BLOCKED_MODULES = {
    "subprocess", "multiprocessing", "ctypes", "requests", "urllib", "socket",
    "http", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "webbrowser", "openai", "litellm", "importlib",
    "anthropic", "ollama", "dotenv", "keyring",
}
_LOCAL_DYNAMIC_CALLS = {"eval", "exec", "compile", "__import__"}
_LOCAL_CLIENT_CALLS = {"OpenAI", "AsyncOpenAI", "OpenAIServerModel"}
_LOCAL_PROCESS_NETWORK_CALLS = {
    "concurrent.futures.ProcessPoolExecutor", "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell", "asyncio.open_connection", "asyncio.start_server",
    "asyncio.open_unix_connection", "asyncio.start_unix_server",
}
_LOCAL_OS_BLOCKED = {
    "environ", "environb", "getenv", "getenvb", "putenv", "unsetenv", "system",
    "popen", "fork", "forkpty", "kill", "killpg", "startfile", "chdir", "fchdir",
    "chroot", "posix_spawn", "posix_spawnp",
}


def _validate_native_local_code(code: str, class_name: str, workspace: Path) -> None:
    """Screen concrete external actions without prescribing a local algorithm.

    The native validator owns dependency availability, provider interfaces and
    runtime behavior. Optional local packages and all normal stdlib algorithms
    reach that validator. This is an experiment policy, not a Python sandbox.
    """
    tree = ast.parse(code)
    if not any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
        raise ValueError("Generated source does not define its declared provider class")
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _LOCAL_BLOCKED_MODULES:
                    raise ValueError(f"Generated provider imports an external-action module: {root}")
                aliases[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in _LOCAL_BLOCKED_MODULES:
                raise ValueError(f"Generated provider imports an external-action module: {module}")
            if any(alias.name == "*" for alias in node.names):
                raise ValueError("Generated provider must name imported operations")
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{module}.{alias.name}"

    def canonical(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                parent = canonical(node.args[0])
                return f"{parent}.{node.args[1].value}" if parent else ""
        dotted = _dotted(node)
        return ".".join([aliases.get(dotted[0], dotted[0]), *dotted[1:]]) if dotted else ""

    def check_operation(name):
        parts = name.split(".")
        if parts[0] in _LOCAL_BLOCKED_MODULES:
            raise ValueError("Generated provider invokes an external network or process operation")
        if parts[0] == "os" and len(parts) > 1 and (
                parts[1] in _LOCAL_OS_BLOCKED or parts[1].startswith(("spawn", "exec"))):
            raise ValueError("Generated provider accesses environment or process state")
        if name in _LOCAL_DYNAMIC_CALLS or name in {f"builtins.{item}" for item in _LOCAL_DYNAMIC_CALLS}:
            raise ValueError("Generated provider executes Python dynamically")
        if name in _LOCAL_PROCESS_NETWORK_CALLS:
            raise ValueError("Generated provider invokes an external network or process operation")

    # These literal defaults only disable implicit model downloads. Exempt the
    # precise call and its receiver, rather than allowing environ generally.
    offline_defaults, offline_imports = set(), set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and canonical(node.func) == "os.environ.setdefault"
                and len(node.args) == 2 and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == "1"):
            offline_defaults.add(node)
            receiver = node.func
            while isinstance(receiver, ast.Attribute):
                offline_defaults.add(receiver)
                receiver = receiver.value
            if isinstance(receiver, ast.Name):
                offline_defaults.add(receiver)
                if aliases.get(receiver.id) == "os.environ":
                    offline_imports.add(receiver.id)

    for node in ast.walk(tree):
        if node in offline_defaults:
            continue
        if isinstance(node, (ast.Name, ast.Attribute)):
            check_operation(canonical(node))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (node.module == "os" and alias.name == "environ"
                        and (alias.asname or alias.name) in offline_imports):
                    continue
                check_operation(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Call):
            # A literal getattr must not expose the environment indirectly.
            check_operation(canonical(node))
            operation = canonical(node.func)
            check_operation(operation)
            if operation.split(".")[-1] in _LOCAL_CLIENT_CALLS:
                raise ValueError("Generated provider constructs a model client instead of using the supplied model")
            # Keep literal file routing under the same native storage tree.
            if operation in {"open", "builtins.open", "pathlib.Path", "Path", "os.makedirs", "os.mkdir"}:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    _storage_path(workspace, node.args[0].value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8"))


def _identifier(value: Any) -> str:
    if (not isinstance(value, str) or not _IDENTIFIER.fullmatch(value)
            or keyword.iskeyword(value) or len(value) >= 100 or value.casefold() in _WINDOWS_RESERVED):
        raise ValueError("Candidate identifiers must be unique Python identifiers shorter than 100 characters")
    return value


def _assignment(module: ast.Module, name: str) -> ast.AST:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                for target in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    raise ValueError(f"Missing literal assignment: {name}")


def _dict_item(node: ast.AST, key: str) -> ast.AST:
    if isinstance(node, ast.Dict):
        for item_key, value in zip(node.keys, node.values):
            if isinstance(item_key, ast.Constant) and item_key.value == key:
                return value
    raise ValueError(f"Missing dictionary item: {key}")


def _registry(workspace: Path) -> tuple[dict[str, str], ast.Dict]:
    tree = ast.parse((workspace / "EvolveLab/memory_types.py").read_text(encoding="utf-8"))
    enums = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MemoryType":
            for field in node.body:
                if isinstance(field, ast.Assign) and len(field.targets) == 1 and isinstance(field.targets[0], ast.Name):
                    value = ast.literal_eval(field.value)
                    if isinstance(value, str):
                        enums[value] = field.targets[0].id
    mapping = _assignment(tree, "PROVIDER_MAPPING")
    if not isinstance(mapping, ast.Dict):
        raise ValueError("PROVIDER_MAPPING must be a dictionary literal")
    return enums, mapping


def _parent_state(workspace: Path, provider_id: str) -> dict[str, Any]:
    enums, mapping = _registry(workspace)
    enum_name = enums.get(provider_id)
    if enum_name is None:
        raise ValueError("Current parent provider is absent from the isolated registry")
    binding = None
    for key, value in zip(mapping.keys, mapping.values):
        if (isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
                and key.value.id == "MemoryType" and key.attr == enum_name):
            binding = ast.literal_eval(value)
            break
    if not isinstance(binding, tuple) or len(binding) != 2:
        raise ValueError("Current parent mapping is not a literal (class, module) pair")
    class_name, module_name = map(_identifier, binding)
    source_path = workspace / "EvolveLab/providers" / f"{module_name}.py"
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("Parent provider source must be a regular isolated file")
    config_tree = ast.parse((workspace / "EvolveLab/config.py").read_text(encoding="utf-8"))
    providers = _dict_item(_assignment(config_tree, "DEFAULT_CONFIG"), "providers")
    config = None
    if isinstance(providers, ast.Dict):
        for key, value in zip(providers.keys, providers.values):
            if (isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name)
                    and key.value.id == "MemoryType" and key.attr == enum_name):
                config = ast.literal_eval(value)
                break
    if not isinstance(config, dict):
        raise ValueError("Parent provider configuration must be a dictionary literal")
    config = deepcopy(config)
    if provider_id == "lightweight_memory":
        # Match configure_provider's effective baseline setting after enabling Phi.
        config["enable_longterm_provision"] = True
    return {"class_name": class_name, "module_name": module_name,
            "source_sha256": _sha(source_path.read_bytes()),
            "code": source_path.read_text(encoding="utf-8"), "config": config}


def _routing_key(key: str) -> bool:
    return key in {"storage", "directory", "filename"} or key.endswith(
        ("_path", "_dir", "_file", "_directory", "_filename", "_cache"))


def _without_routing(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items()
            if key not in _META_KEYS and not key.startswith("philo_orientation_") and not _routing_key(key)}


def _is_path_text(value: str) -> bool:
    return ("/" in value or "\\" in value or value.startswith(".")
            or bool(PureWindowsPath(value).drive))


def _storage_path(workspace: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Writable config paths must be nonempty strings")
    if os.name != "nt" and PureWindowsPath(value).drive:
        raise ValueError("Writable config path escapes workspace/storage")
    path = Path(value.replace("\\", "/"))
    target = (path if path.is_absolute() else workspace / path).resolve()
    storage = (workspace / "storage").resolve()
    if not target.is_relative_to(storage):
        raise ValueError("Writable config path escapes workspace/storage")
    return target


def _frozen_config(workspace: Path, parent: dict[str, Any], alias: str) -> dict[str, Any]:
    result = {key: deepcopy(value) for key, value in parent.items()
              if key not in _META_KEYS and not key.startswith("philo_orientation_")}
    storage = (workspace / "storage").resolve()
    for key, value in list(result.items()):
        if _routing_key(key) and isinstance(value, str) and _is_path_text(value):
            original = _storage_path(workspace, value).relative_to(storage)
            # Remove the old provider namespace, preserving nested filenames.
            suffix = original.parts[1:] if original.parts else ()
            result[key] = "./storage/" + "/".join((alias, *suffix))
    result["storage_dir"] = f"./storage/{alias}"
    if _without_routing(result) != _without_routing(parent):
        raise ValueError("Frozen candidate changed effective non-routing configuration")
    return result


def _parse_config(config: dict[str, Any]) -> dict[str, Any]:
    """Match the creator's literal parsing without executing config expressions."""
    result = {}
    for key, value in config.items():
        _identifier(key)
        if isinstance(value, str):
            value = re.sub(r"\s*\([^)]*\)", "", value.replace("`", "")).strip()
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            else:
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
        result[key] = value
    return result


def _generated_config_fields(proposal: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Separate declared settings from bold labels inside generated source.

    Native parsing gathers every ``**label**: value`` in the response, including
    provider docstrings. Prefer its explicit Configuration section; older native
    responses without that heading retain their settings outside code fences.
    The literal/value and storage validations still run after this extraction.
    """
    raw = proposal.get("_raw_llm_output")
    if not isinstance(raw, str):
        return deepcopy(proposal["config_updates"]), None
    outside, fence = [], None
    for line_number, line in enumerate(raw.splitlines(), 1):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            symbol = marker[1][0]
            if fence is None:
                fence = (symbol, len(marker[1]))
            elif symbol == fence[0] and len(marker[1]) >= fence[1]:
                fence = None
            continue
        if fence is None:
            outside.append((line_number, line))
    headings = []
    for index, (_, line) in enumerate(outside):
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            title = match[2].strip(" *`_").casefold()
            if title in {"configuration", "configuration parameters", "configuration updates", "配置", "配置参数"}:
                headings.append((index, len(match[1])))
    if len(headings) > 1:
        raise ValueError("Generated response has multiple Configuration sections")
    selected, method = outside, "outside_code_fences"
    if headings:
        start, depth = headings[0]
        end = len(outside)
        for index in range(start + 1, len(outside)):
            match = re.match(r"^\s{0,3}(#{1,6})\s+", outside[index][1])
            if match and len(match[1]) <= depth:
                end = index
                break
        selected, method = outside[start + 1:end], "explicit_configuration_section"
    fields, sources = {}, {}
    metadata = {"Class Name", "Module Name", "Enum Name", "Enum Value"}
    for line_number, line in selected:
        match = re.match(r"^\s*(?:[-+]\s+)?\*\*([^*]+)\*\*\s*:\s*(.+?)\s*$", line)
        if match is None:
            continue
        key, value = match[1].strip(), match[2].strip()
        if key in metadata:
            continue
        if not headings and not _IDENTIFIER.fullmatch(key):
            continue
        _identifier(key)
        if key in fields and fields[key] != value:
            raise ValueError(f"Generated Configuration repeats a setting with conflicting values: {key}")
        fields[key] = value
        sources.setdefault(key, []).append(line_number)
    if not fields and proposal.get("config_updates"):
        raise ValueError("Generated response has no configuration fields outside provider source")
    evidence = {"method": method, "raw_response_sha256": _sha(raw.encode("utf-8")),
                "retained_keys": list(fields), "source_lines": sources,
                "discarded_native_keys": sorted(set(proposal["config_updates"]) - set(fields))}
    return fields, evidence


def _validate_config_paths(workspace: Path, config: dict[str, Any]) -> None:
    for key, value in config.items():
        if key in _META_KEYS:
            continue
        if _routing_key(key):
            if not isinstance(value, str):
                raise ValueError("Writable config paths must be strings")
            # Plain filenames are resolved by the provider under storage_dir.
            if _is_path_text(value):
                _storage_path(workspace, value)
        elif isinstance(value, dict):
            _validate_config_paths(workspace, value)


def _reserved_candidate_names(workspace: Path, work_dir: Path, exclude_config=None):
    enums, mapping = _registry(workspace)
    names = {"class_name": set(), "module_name": set(),
             "enum_name": set(enums.values()), "enum_value": set(enums)}
    sources = []
    for node in mapping.values:
        pair = ast.literal_eval(node)
        names["class_name"].add(pair[0])
        names["module_name"].add(pair[1])
    names["module_name"].update(path.stem for path in (workspace / "EvolveLab/providers").glob("*.py"))
    storage = workspace / "storage"
    storage_names = {path.name for path in storage.iterdir()} if storage.exists() else set()
    excluded = Path(exclude_config).resolve() if exclude_config else None
    for path in sorted(work_dir.glob("generated_system_*.json")):
        if path.resolve() == excluded:
            continue
        previous = json.loads(path.read_text(encoding="utf-8"))
        for field in ("class_name", "module_name"):
            names[field].add(previous["provider_code"][field])
        for field in ("enum_name", "enum_value"):
            names[field].add(previous["memory_type_info"][field])
        for key, value in previous.get("config_updates", {}).items():
            if _routing_key(key) and isinstance(value, str) and _is_path_text(value):
                parts = _storage_path(workspace, value).relative_to(storage.resolve()).parts
                if parts:
                    storage_names.add(parts[0])
        sources.append(str(path.resolve()))
    return names, storage_names, sources


def _replace_storage_namespace(value, replacements):
    for old, new in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        components = r"[/\\]".join(re.escape(part) for part in old.split("/"))
        pattern = r"(?P<prefix>(?:^|[/\\])storage(?P<sep>[/\\]))" + components + r"(?=$|[/\\])"
        changed, count = re.subn(pattern, lambda match: match["prefix"] + new.replace("/", match["sep"]), value)
        if count:
            return changed
    return value


class _NamespaceRename(ast.NodeTransformer):
    def __init__(self, symbols, literals, storage):
        self.symbols, self.literals, self.storage = symbols, literals, storage

    def visit_Name(self, node):
        node.id = self.symbols.get(node.id, node.id)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = self.symbols.get(node.attr, node.attr)
        return node

    def visit_ClassDef(self, node):
        node.name = self.symbols.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.name = self.symbols.get(node.name, node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node):
        node.arg = self.symbols.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node.arg = self.symbols.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.literals.get(node.value, _replace_storage_namespace(node.value, self.storage))
        return node

    def visit_alias(self, node):
        node.name = ".".join(self.symbols.get(part, part) for part in node.name.split("."))
        node.asname = self.symbols.get(node.asname, node.asname)
        return node

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = ".".join(self.symbols.get(part, part) for part in node.module.split("."))
        return self.generic_visit(node)

    def visit_Global(self, node):
        node.names = [self.symbols.get(name, name) for name in node.names]
        return node

    visit_Nonlocal = visit_Global


def project_candidate_namespace(config, workspace, *, work_dir, job_id, exclude_config=None):
    """Resolve namespace collisions without changing the generated algorithm.

    ``exclude_config`` identifies the paid candidate being repaired in place;
    all other native generated files reserve names even before creation. This
    function is pure: callers save the returned config and audit evidence.
    """
    workspace, work_dir = Path(workspace).resolve(), Path(work_dir).resolve()
    result = deepcopy(config)
    provider, metadata = result["provider_code"], result["memory_type_info"]
    original = {**{key: provider[key] for key in ("class_name", "module_name")},
                **{key: metadata[key] for key in ("enum_name", "enum_value")}}
    for value in original.values():
        _identifier(value)
    reserved, storage_names, sources = _reserved_candidate_names(workspace, work_dir, exclude_config)
    storage_root = (workspace / "storage").resolve()
    routed_roots = set()

    def find_storage(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if _routing_key(key) and isinstance(child, str) and _is_path_text(child):
                    parts = _storage_path(workspace, child).relative_to(storage_root).parts
                    if parts:
                        routed_roots.add(parts[0])
                elif isinstance(child, dict):
                    find_storage(child)
    find_storage(result.get("config_updates", {}))
    collisions = [key for key, value in original.items() if value in reserved[key]]
    if routed_roots & storage_names:
        collisions.append("storage_namespace")
    if not collisions:
        return result, None
    code = provider["code"]
    relative = work_dir.relative_to(workspace.parent).as_posix()
    seed = {"job_id": job_id, "round_dir": relative, "code_sha256": _sha(code.encode("utf-8")),
            "original": original, "reserved": {key: sorted(value) for key, value in reserved.items()},
            "storage_names": sorted(storage_names)}
    for counter in range(100):
        suffix = "_hss_" + _json_sha({**seed, "counter": counter})[:12]
        renamed = {key: _identifier(value[:99 - len(suffix)] + suffix) for key, value in original.items()}
        if (all(value not in reserved[key] for key, value in renamed.items())
                and renamed["enum_value"] not in storage_names):
            break
    else:
        raise ValueError("Could not reserve a unique generated provider namespace")
    literal_map = {}
    for key, old in original.items():
        if old in literal_map and literal_map[old] != renamed[key]:
            raise ValueError("Generated namespace cannot be renamed unambiguously")
        literal_map[old] = renamed[key]
    symbols = {old: literal_map[old] for old in (original["class_name"], original["module_name"], original["enum_name"])}
    main_storage = result.get("config_updates", {}).get("storage_dir")
    main_parts = (_storage_path(workspace, main_storage).relative_to(storage_root).parts
                  if isinstance(main_storage, str) else ())
    main_alias = main_parts[0] if main_parts else original["enum_value"]
    storage_map = {old: renamed["enum_value"] if old == main_alias
                   else f"{renamed['enum_value']}/{old}" for old in routed_roots | {original["enum_value"]}}
    before = ast.parse(code)
    after = _NamespaceRename(symbols, literal_map, storage_map).visit(deepcopy(before))
    rewritten = ast.unparse(ast.fix_missing_locations(after)) + "\n"
    inverse_literals = {new: old for old, new in literal_map.items()}
    inverse_symbols = {new: old for old, new in symbols.items()}
    inverse_storage = {new: old for old, new in storage_map.items()}
    restored = _NamespaceRename(inverse_symbols, inverse_literals, inverse_storage).visit(ast.parse(rewritten))
    if ast.dump(restored, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise ValueError("Generated namespace rewrite could not preserve the provider algorithm")
    provider.update(class_name=renamed["class_name"], module_name=renamed["module_name"], code=rewritten)
    metadata.update(enum_name=renamed["enum_name"], enum_value=renamed["enum_value"])

    def rewrite_config(value):
        if isinstance(value, dict):
            return {key: rewrite_config(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite_config(child) for child in value]
        if isinstance(value, str):
            return literal_map.get(value, _replace_storage_namespace(value, storage_map))
        return value
    result["config_updates"] = rewrite_config(result["config_updates"])
    evidence = {"method": "deterministic_namespace_projection", "collisions": collisions,
                "identifier_map": {key: {"before": original[key], "after": renamed[key]} for key in original},
                "storage_namespace_map": storage_map, "pending_config_sources": sources,
                "original_code_sha256": _sha(code.encode("utf-8")),
                "projected_code_sha256": _sha(rewritten.encode("utf-8")),
                "algorithm_ast_preserved_under_inverse_namespace_map": True,
                "orientation_unchanged": result.get("orientation_items") == config.get("orientation_items")}
    result["_namespace_projection"] = evidence
    return result, evidence


def _dotted(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_dotted(node.value), node.attr]
    return []


def _validate_generated_code(code: str, class_name: str, workspace: Path) -> None:
    tree = ast.parse(code)
    if not any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
        raise ValueError("Generated source does not define its declared provider class")
    os_aliases = set()
    path_aliases = set()
    makedirs_aliases = {"makedirs"}
    response_getattrs = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "getattr" and len(node.args) in (2, 3) and not node.keywords
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
        and node.args[1].value in {"content", "reasoning_content", "tool_calls"}
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if (root not in _SAFE_STDLIB or
                        ("." in alias.name and alias.name not in _SAFE_SUBMODULES)):
                    raise ValueError(f"Generated provider imports a disallowed module: {root}")
                if root == "os":
                    if alias.name not in {"os", "os.path"}:
                        raise ValueError("Generated provider may only use os.path and os.makedirs")
                    if alias.name == "os.path" and alias.asname:
                        path_aliases.add(alias.asname)
                    else:
                        os_aliases.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or (module not in _NATIVE_IMPORTS and (
                    module.split(".")[0] not in _SAFE_STDLIB
                    or ("." in module and module not in _SAFE_SUBMODULES))):
                raise ValueError(f"Generated provider imports a disallowed module: {module}")
            if any(alias.name == "*" or alias.name in _FORBIDDEN_NAMES for alias in node.names):
                raise ValueError("Generated provider uses a disallowed import")
            if module == "os" and any(alias.name not in {"path", "makedirs"} for alias in node.names):
                raise ValueError("Generated provider may only use os.path and os.makedirs")
            if module == "os":
                for alias in node.names:
                    if alias.name == "path":
                        path_aliases.add(alias.asname or alias.name)
                    elif alias.name == "makedirs":
                        makedirs_aliases.add(alias.asname or alias.name)
            if module == "logging" and any(alias.name in {"handlers", "config"} for alias in node.names):
                raise ValueError("Generated provider may not load logging network or file handlers")
            if module.startswith("os.") and (module != "os.path" or any(
                    alias.name not in _OS_PATH_METHODS for alias in node.names)):
                raise ValueError("Generated provider uses a disallowed os.path operation")
        if (isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES
                and id(node) not in response_getattrs):
            raise ValueError("Generated provider uses a disallowed dynamic or network operation")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            normal_super_init = (
                node.attr == "__init__" and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "super"
                and not node.value.args and not node.value.keywords
            )
            if not normal_super_init:
                raise ValueError("Generated provider uses a disallowed reflective attribute")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name in _FORBIDDEN_CALLS:
                raise ValueError("Generated provider constructs a client or invokes a disallowed action")
            if name in {"open", "Path", *makedirs_aliases} and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    _storage_path(workspace, node.args[0].value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted and dotted[0] in os_aliases:
                allowed = (len(dotted) == 2 and dotted[1] in {"path", "makedirs"}) or (
                    len(dotted) == 3 and dotted[1] == "path" and dotted[2] in _OS_PATH_METHODS)
                if not allowed:
                    raise ValueError("Generated provider may only use os.path and os.makedirs")
            if dotted and dotted[0] in path_aliases and (
                    len(dotted) != 2 or dotted[1] not in _OS_PATH_METHODS):
                raise ValueError("Generated provider uses a disallowed os.path operation")


def _alias(job: dict[str, Any], work_dir: Path, run_dir: Path) -> str:
    relative = work_dir.resolve().relative_to(run_dir).as_posix()
    identity = f"{job['job_id']}\0{relative}"
    label = re.sub(r"[^A-Za-z0-9_]", "_", job["job_id"])[:42]
    return _identifier(f"philo_{label}_{_sha(identity.encode('utf-8'))[:14]}")


def install_native_policy(run_dir: str | Path, job: dict[str, Any], *, compact_provider=True,
                          generated_code_policy="restricted") -> Callable[[], None]:
    """Install reversible wrappers after activating one isolated native workspace.

    Repeated installation for the same job returns the same restoration callback;
    another active job is rejected. Run actor/candidate tasks in fresh processes
    as required by the native registry's import-time caches.
    """
    run_dir = Path(run_dir).resolve(strict=True)
    workspace = (run_dir / "workspace").resolve(strict=True)
    if any(type(job.get(key)) is not bool for key in ("code_evolve", "orientation_evolve")):
        raise ValueError("Ablation code and orientation switches must be booleans")
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise ValueError("Ablation job_id is required")
    if generated_code_policy not in {"restricted", "native_local"}:
        raise ValueError("Unknown generated provider policy")
    job = deepcopy(job)
    module = importlib.import_module("MemEvolve.phases.phase_generator")
    module_path = Path(module.__file__).resolve(strict=True)
    if not module_path.is_relative_to(workspace):
        raise ValueError("Activate the isolated native workspace before installing its policy")
    generator_class = module.PhaseGenerator
    current = getattr(generator_class, "_philo_ablation_policy", None)
    identity = (str(run_dir), job["job_id"], job["code_evolve"], job["orientation_evolve"],
                compact_provider, generated_code_policy)
    if current is not None:
        if current["identity"] != identity:
            raise ValueError("A different ablation policy is already installed")
        return current["restore"]
    original_build = generator_class._build_generation_prompt
    original_generate = generator_class._generate_single_system
    original_parse = getattr(generator_class, "_parse_system_config", None)

    @functools.wraps(original_parse)
    def parse(self, response_text):
        config = original_parse(self, response_text)
        if config is not None:
            return config
        from .hss_generation_format import recover_missing_registration
        return recover_missing_registration(response_text,
            lambda text: original_parse(self, text))

    @functools.wraps(original_build)
    def build(self, *args, **kwargs):
        prompt = original_build(self, *args, **kwargs)
        scope = ("Generate a compact provider, preferably at most 220 Python lines, "
                 if compact_provider else
                 "Generate a complete memory provider suited to the observed tasks. "
                 "Choose the architecture and code length required by its design; "
                 "there is no experimental preference for a short provider. ")
        local_libraries = (
            "Use normal Python stdlib and available local algorithm libraries as needed. "
            "Optional dependencies must have a local fallback when unavailable; do not install "
            "packages or download models/resources. Use atomic writes, locks and local numerical "
            "methods when useful; the native validator checks dependency and provider behavior. "
            if generated_code_policy == "native_local" else
            "Use Python stdlib and EvolveLab.base_memory / EvolveLab.memory_types. ")
        local_storage = ("Keep local persistence under the provider's storage directory. "
                         if generated_code_policy == "native_local" else
                         "Use os.path, os.makedirs and JSON files for local storage. ")
        pending_names = ""
        if generated_code_policy == "native_local":
            reserved, storage_names, sources = _reserved_candidate_names(workspace, Path(self.work_dir))
            if sources:
                pending_names = ("\nProvider namespaces already reserved by pending candidates in this round "
                    "must also be unique (these candidates may not yet be in MemoryType): "
                    + json.dumps({key: sorted(value) for key, value in reserved.items()}, ensure_ascii=False)
                    + "; storage namespaces: " + json.dumps(sorted(storage_names), ensure_ascii=False))
        return prompt + (
            "\n\nExperiment candidate constraints: " + scope +
            local_libraries +
            "Use the supplied config.get('model') for all model calls; do not construct clients, use "
            "network APIs, download resources, execute generated Python dynamically, or run shell commands. "
            "Use one unique ./storage/<your enum_value>/ directory for every writable file and config path. " +
            local_storage + "Accept the current principles from "
            "config['philo_orientation_items'] and the shared request/trajectory metadata; do not hardcode "
            "initial principles or write lineage files. Based on observed development evidence you may "
            "retain, revise, or clear the current principles. Omit orientation_items to inherit unchanged, "
            "or emit **orientation_items**: followed by a single-line JSON list; [] clears them. "
            "A code or principle change is not required. Choose new valid unique Python identifiers "
            "for the provider class, module, enum name and enum value." + pending_names
        )

    @functools.wraps(original_generate)
    def generate(self, *args, **kwargs):
        proposal = original_generate(self, *args, **kwargs)
        if proposal is None:
            return None
        config = deepcopy(proposal)
        snapshot = getattr(self, "_orientation_snapshot", None)
        if snapshot is None or snapshot.provider_id != self.default_provider:
            raise ValueError("Candidate projection requires the current parent orientation snapshot")
        assert_unchanged(snapshot)
        if snapshot.path.parent.parent.resolve() != (run_dir / "lineage").resolve():
            raise ValueError("Candidate parent belongs to another ablation lineage")
        parent = _parent_state(workspace, snapshot.provider_id)
        proposed_items = config.get("orientation_items", list(snapshot.items))
        if job["orientation_evolve"] and (not isinstance(proposed_items, list)
                or any(not isinstance(item, str) or not item.strip() for item in proposed_items)):
            raise ValueError("Proposed orientation must be a list of nonempty strings or []")
        if not job["orientation_evolve"]:
            config["orientation_items"] = list(snapshot.items)
        child_items = config.get("orientation_items", list(snapshot.items))
        if not job["code_evolve"]:
            alias = _alias(job, Path(self.work_dir), run_dir)
            config["provider_code"] = {key: parent[key] for key in ("class_name", "module_name", "code")}
            config["_philo_freeze_code"] = {key: parent[key]
                                            for key in ("module_name", "class_name", "source_sha256")}
            config["memory_type_info"] = {"enum_name": alias.upper(), "enum_value": alias}
            config["config_updates"] = _frozen_config(workspace, parent["config"], alias)
            child_code_hash = parent["source_sha256"]
        else:
            config.pop("_philo_freeze_code", None)
            provider = config["provider_code"]
            metadata = config["memory_type_info"]
            for value in (provider["class_name"], provider["module_name"], metadata["enum_name"], metadata["enum_value"]):
                _identifier(value)
            raw_fields, extraction = _generated_config_fields(config)
            config["config_updates"] = _parse_config(raw_fields)
            if extraction is not None:
                config["_config_extraction"] = extraction
            config["config_updates"] = {
                key: value for key, value in config["config_updates"].items()
                if key not in _META_KEYS and not key.startswith("philo_orientation_")
            }
            _validate_config_paths(workspace, config["config_updates"])
            if generated_code_policy == "native_local":
                config, _ = project_candidate_namespace(config, workspace,
                    work_dir=Path(self.work_dir), job_id=job["job_id"])
                provider, metadata = config["provider_code"], config["memory_type_info"]
                _validate_config_paths(workspace, config["config_updates"])
            alias = metadata["enum_value"]
            if (workspace / "EvolveLab/providers" / f"{provider['module_name']}.py").exists():
                raise ValueError("Generated provider module already exists; candidate was not overwritten")
            validate_code = (_validate_native_local_code if generated_code_policy == "native_local"
                             else _validate_generated_code)
            validate_code(provider["code"], provider["class_name"], workspace)
            encoded = provider["code"].replace("\n", os.linesep).encode("utf-8")
            child_code_hash = _sha(encoded)
        enums, _ = _registry(workspace)
        if alias in enums or config["memory_type_info"]["enum_name"] in enums.values():
            raise ValueError("Candidate enum already exists; candidate was not overwritten")
        # This metadata always refers to the verified current parent, including
        # the second round and aliases whose provider class is still the baseline.
        config["_orientation_parent"] = {
            "lineage_root": str((run_dir / "lineage").resolve()), "snapshot": snapshot.to_dict(),
        }
        append_event(run_dir / "evidence/factor_projection.jsonl", {
            "event": "factor_projection", "job_id": job["job_id"],
            "parent_provider": snapshot.provider_id, "candidate_provider": alias,
            "code_evolve": job["code_evolve"], "orientation_evolve": job["orientation_evolve"],
            "parent_items_sha256": _json_sha(list(snapshot.items)),
            "proposal_items_sha256": _json_sha(proposed_items),
            "child_items_sha256": _json_sha(child_items),
            "orientation_proposal_overridden": proposed_items != child_items,
            "code_proposal_overridden": proposal.get("provider_code") != config["provider_code"],
            "config_proposal_overridden": proposal.get("config_updates") != config["config_updates"],
            "parent_code_sha256": parent["source_sha256"], "child_code_sha256": child_code_hash,
            "child_code_hash_scope": "expected_creator_file_bytes_before_creation",
            "parent_effective_config_sha256": _json_sha(_without_routing(parent["config"])),
            "child_effective_config_sha256": _json_sha(_without_routing(config["config_updates"])),
            "ordinary_memory_learning": True,
            "generated_code_policy": generated_code_policy,
        })
        return config

    def restore():
        if generator_class._build_generation_prompt is build:
            generator_class._build_generation_prompt = original_build
        if generator_class._generate_single_system is generate:
            generator_class._generate_single_system = original_generate
        if (generated_code_policy == "native_local" and original_parse is not None
                and generator_class._parse_system_config is parse):
            generator_class._parse_system_config = original_parse
        active = getattr(generator_class, "_philo_ablation_policy", None)
        if active is not None and active["identity"] == identity:
            delattr(generator_class, "_philo_ablation_policy")

    generator_class._build_generation_prompt = build
    generator_class._generate_single_system = generate
    if generated_code_policy == "native_local" and original_parse is not None:
        generator_class._parse_system_config = parse
    generator_class._philo_ablation_policy = {"identity": identity, "restore": restore}
    return restore
