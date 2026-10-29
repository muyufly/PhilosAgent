"""Present JSON text-memory values at the native string guidance interface."""

from dataclasses import replace
from functools import wraps
import hashlib
import json
import math

from .ablation_runtime import digest


def _json_value(value, active):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Text memory contains a non-finite JSON number")
        return
    if type(value) not in (dict, list):
        raise TypeError("Text memory must contain a string or a JSON value")
    if id(value) in active:
        raise ValueError("Text memory contains a cyclic JSON structure")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("JSON text-memory objects require string keys")
            children = value.values()
        else:
            children = value
        for child in children:
            _json_value(child, active)
    finally:
        active.remove(id(value))


def normalize_memory_response(response):
    """Return a copied native response only when JSON text needs presentation.

    API/tool memories retain their native objects. Strings retain their exact
    content. No retrieval, ranking, memory storage or philosophy is modified.
    """
    memories, evidence = [], []
    for item in response.memories:
        kind = getattr(item.type, "value", item.type)
        if kind != "text" or isinstance(item.content, str):
            memories.append(item)
            continue
        value = item.content
        _json_value(value, set())
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        memories.append(replace(item, content=text))
        evidence.append({"memory_id": item.id, "input_type": type(value).__name__,
                         "input_json_sha256": digest(value),
                         "output_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "representation": "lossless_json_text"})
    return (replace(response, memories=memories), evidence) if evidence else (response, [])


def guard_memory_response(original):
    """Wrap the shared provider interface; keep native memory operations intact."""
    @wraps(original)
    def provide(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        converted, evidence = normalize_memory_response(response)
        if evidence:
            self._record("memory_content_format", purpose="native_text_interface_format",
                         algorithm_changed=False, adapter_mutates_provider_state=False,
                         items=evidence)
        return converted
    return provide
