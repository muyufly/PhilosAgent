"""Lossless native action decoding at the HSS tool-call boundary.

Native steps accept dictionaries, but permissive JSON repair can return strings
for a provider's PLAN/DSML output. Decode declared actions before that boundary;
malformed turns become ordinary observations in the same actor trajectory.
"""

from __future__ import annotations

from copy import copy
from functools import wraps
import hashlib
import html
import json
import re
from unittest.mock import patch


class InvalidToolResponse(ValueError):
    pass


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidToolResponse("Duplicate JSON fields are ambiguous")
        result[key] = value
    return result


def _strict_json(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def _action(value, allowed):
    if not isinstance(value, dict):
        raise InvalidToolResponse("Each entry in tools must be an object")
    name, arguments = value.get("name"), value.get("arguments")
    if not isinstance(name, str) or name not in allowed:
        raise InvalidToolResponse("Action must name one of the declared tools")
    if isinstance(arguments, str):
        try:
            arguments = _strict_json(arguments)
        except ValueError:
            raise InvalidToolResponse("Tool arguments must be a JSON object") from None
    if not isinstance(arguments, dict):
        raise InvalidToolResponse("Tool arguments must be a JSON object")
    if name == "final_answer" and (not isinstance(arguments.get("answer"), str) or not arguments["answer"].strip()):
        raise InvalidToolResponse("final_answer requires a nonempty answer string")
    result = {"name": name, "arguments": arguments}
    if isinstance(value.get("id"), str):
        result["id"] = value["id"]
    return result


def _final_action(value, allowed):
    if "final_answer" not in allowed or not isinstance(value, dict):
        raise InvalidToolResponse("Recovery requires a declared final_answer action")
    actions = value.get("tools")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        raise InvalidToolResponse("Recovery requires exactly one final_answer action")
    action = actions[0]
    if action.get("name") != "final_answer":
        raise InvalidToolResponse("Recovery requires a declared final_answer action")
    return action


def _decode_complete_string_body(body):
    """Decode escapes once while treating internal unescaped quotes as text."""
    escaped, index = [], 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            if index + 1 >= len(body):
                raise InvalidToolResponse("Final answer ends in an unfinished escape")
            escaped.append(body[index:index + 2])
            index += 2
            continue
        if char == '"':
            escaped.append('\\"')
        elif ord(char) < 0x20:
            escaped.append(json.dumps(char)[1:-1])
        else:
            escaped.append(char)
        index += 1
    try:
        return json.loads('"' + "".join(escaped) + '"')
    except ValueError:
        raise InvalidToolResponse("Final answer contains an invalid string escape") from None


def recover_final_answer(message, allowed):
    """Recover one whole explicit final answer, with its exact source span.

    Only a single final action in a complete envelope is eligible. For malformed
    internal quotes, replacing the entire answer span with a sentinel must leave
    strict, duplicate-free JSON. Every character in that span is then decoded;
    permissive JSON repair never chooses a shorter report prefix.
    """
    raw = _field(message, "content")
    if not isinstance(raw, str):
        raise InvalidToolResponse("Final answer recovery requires textual native JSON")
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if fence:
        text = fence[1]
    try:
        value = _strict_json(text)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        action = _final_action(value, allowed)
        arguments = action.get("arguments")
        try:
            canonical = _action(action, allowed)
            method = "strict_json_final_answer"
        except InvalidToolResponse:
            if not isinstance(arguments, str) or not arguments.strip():
                raise
            canonical = _action({**action, "arguments": {"answer": arguments}}, allowed)
            method = "strict_json_bare_arguments_string"
        answer = canonical["arguments"]["answer"]
        return {"think": value.get("think", ""), "tools": [canonical]}, {
            "method": method, "lossless": True, "decoded_answer_chars": len(answer),
            "source_content_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "decoded_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest()}

    # With broken quoting, a second apparent action could be either another
    # executable tool or quoted report material. Require the actor to clarify
    # that ambiguity instead of absorbing an action into the report string.
    if len(re.findall(r'"name"\s*:\s*"', text)) != 1:
        raise InvalidToolResponse("Malformed final answer has ambiguous action boundaries")
    sentinel, matches = "__HSS_COMPLETE_FINAL_ANSWER_SPAN__", []
    for key, tail in (("arguments", r'"\s*}\s*]\s*}\s*\Z'),
                      ("answer", r'"\s*}\s*}\s*]\s*}\s*\Z')):
        end = re.search(tail, text)
        if end is None:
            continue
        for start in re.finditer(r'"' + key + r'"\s*:\s*"', text):
            opening, closing = start.end() - 1, end.start()
            if opening >= closing:
                continue
            skeleton = text[:opening] + json.dumps(sentinel) + text[closing + 1:]
            try:
                envelope = _strict_json(skeleton)
                action = _final_action(envelope, allowed)
            except (ValueError, TypeError):
                continue
            expected = sentinel if key == "arguments" else {"answer": sentinel}
            if action.get("arguments") != expected:
                continue
            body = text[opening + 1:closing]
            answer = _decode_complete_string_body(body)
            if not answer.strip():
                continue
            canonical = _action({**action, "arguments": {"answer": answer}}, allowed)
            matches.append(({"think": envelope.get("think", ""), "tools": [canonical]}, {
                "method": "complete_final_string_span", "lossless": True,
                "source_field": key, "source_body_span": [opening + 1, closing],
                "source_body_span_basis": "stripped JSON content without markdown fence",
                "source_content_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "source_body_chars": len(body),
                "source_body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "decoded_answer_chars": len(answer),
                "decoded_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "escape_decoding": "JSON string escapes once; ordinary quotes preserved"}))
    if len(matches) != 1:
        raise InvalidToolResponse("No unique complete final-answer string can be recovered losslessly")
    return matches[0]


_DSML = r"[|｜]*DSML[|｜]*\s*"
_INVOKE = re.compile(r"<" + _DSML + r"invoke\s+([^<>]*?)>([\s\S]*?)</" + _DSML + r"invoke\s*>")
_PARAMETER = re.compile(r"<" + _DSML + r"parameter\s+([^<>]*?)>([\s\S]*?)</" + _DSML + r"parameter\s*>")
_ATTRIBUTE = re.compile(r'''([a-zA-Z_][\w-]*)\s*=\s*(["'])(.*?)\2''')


def _attributes(value):
    pairs = [(match[0], html.unescape(match[2])) for match in _ATTRIBUTE.findall(value)]
    if len({name for name, _ in pairs}) != len(pairs):
        raise InvalidToolResponse("Duplicate DSML attributes")
    return dict(pairs)


def _dsml_actions(text, allowed):
    matches = list(_INVOKE.finditer(text))
    if not matches or len(matches) != len(re.findall(r"<" + _DSML + r"invoke\b", text)):
        raise InvalidToolResponse("Incomplete DSML invocation")
    actions = []
    for match in matches:
        attributes = _attributes(match.group(1))
        parameters = list(_PARAMETER.finditer(match.group(2)))
        residue = _PARAMETER.sub("", match.group(2)).strip()
        if residue or not parameters:
            raise InvalidToolResponse("Incomplete DSML parameters")
        arguments = {}
        for parameter in parameters:
            attrs = _attributes(parameter.group(1))
            name = attrs.get("name")
            if not name or name in arguments:
                raise InvalidToolResponse("Missing or duplicate DSML parameter")
            raw = html.unescape(parameter.group(2))
            if attrs.get("string") == "true":
                value = raw
            elif attrs.get("string") == "false":
                try:
                    value = _strict_json(raw)
                except ValueError:
                    raise InvalidToolResponse("DSML parameter is not valid JSON") from None
            else:
                raise InvalidToolResponse("DSML parameter requires an explicit string type")
            arguments[name] = value
        if set(arguments) == {"arguments"}:
            arguments = arguments["arguments"]
        actions.append(_action({"name": attributes.get("name"), "arguments": arguments}, allowed))
    return actions


def decode_tool_response(message, allowed):
    """Return canonical native JSON plus raw evidence; never invent an action."""
    content = _field(message, "content")
    raw = content if isinstance(content, str) else ""
    evidence = {"original_content": content, "original_content_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "decoder_version": 2, "status": "valid", "format": "native_json"}
    try:
        structured = _field(message, "tool_calls")
        if structured:
            actions = []
            for call in structured:
                function = _field(call, "function")
                actions.append(_action({"name": _field(function, "name"),
                                        "arguments": _field(function, "arguments"),
                                        "id": _field(call, "id")}, allowed))
            evidence.update(status="normalized", format="openai_tool_calls")
            return {"think": raw, "tools": actions}, evidence
        if re.search(r"<" + _DSML + r"invoke\b", raw):
            actions = _dsml_actions(raw, allowed)
            evidence.update(status="normalized", format="dsml")
            return {"think": raw.split("<", 1)[0].strip(), "tools": actions}, evidence
        text = raw.strip()
        fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
        if fence:
            text = fence[1]
            evidence["status"] = "normalized"
        permissive_repair = False
        try:
            value = _strict_json(text)
        except json.JSONDecodeError:
            if re.search(r'"name"\s*:\s*"final_answer"', text):
                normalized, recovery = recover_final_answer({"content": text}, allowed)
                evidence.update(status="normalized", final_answer_recovery=recovery)
                return normalized, evidence
            # Preserve native support for harmless JSON punctuation errors, but
            # validate every resulting action before it can reach execution.
            import json_repair
            value = json_repair.loads(text)
            permissive_repair = True
            evidence["status"] = "normalized"
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], dict) and "tools" in value[0]:
                value = value[0]
            else:
                value = {"tools": value}
            evidence["status"] = "normalized"
        if not isinstance(value, dict):
            raise InvalidToolResponse("Output must be an object with a tools list")
        actions = value.get("tools")
        if isinstance(actions, dict):
            actions = [actions]
            evidence["status"] = "normalized"
        if not isinstance(actions, list) or not actions or len(actions) > 5:
            raise InvalidToolResponse("Output must contain one to five tools")
        if any(isinstance(action, dict) and action.get("name") == "final_answer" for action in actions):
            # Even a syntactically valid bare arguments string must preserve its
            # whole text; never ask json_repair to infer its final-answer body.
            if permissive_repair:
                raise InvalidToolResponse("Malformed final answer cannot use permissive JSON repair")
            if len(actions) != 1:
                return {"think": value.get("think", ""),
                        "tools": [_action(action, allowed) for action in actions]}, evidence
            normalized, recovery = recover_final_answer({"content": json.dumps(value, ensure_ascii=False)}, allowed)
            if recovery["method"] != "strict_json_final_answer":
                evidence.update(status="normalized", final_answer_recovery=recovery)
            return normalized, evidence
        actions = [_action(action, allowed) for action in actions]
        return {"think": value.get("think", ""), "tools": actions}, evidence
    except (InvalidToolResponse, TypeError, ValueError, IndexError) as exc:
        evidence.update(status="invalid", feedback=(
            "Tool output format error: " + str(exc) + ". Continue the same task using "
            '{"think":"...","tools":[{"name":"declared_tool","arguments":{...}}]}. '
            "Call final_answer with the complete requested work when ready; PLAN and Analysis are not tools."))
        return {"think": raw, "tools": []}, evidence


class _StepModel:
    def __init__(self, model, actor, memory_step):
        self._model, self._actor, self._step = model, actor, memory_step

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        response = self._model(*args, **kwargs)
        normalized, evidence = decode_tool_response(response, set(self._actor.tools))
        evidence["step_number"] = self._actor.step_number
        self._actor._hss_tool_decoding_events.append(evidence)
        observer = getattr(self._actor, "_hss_decoding_observer", None)
        if observer:
            observer(evidence)
        if evidence["status"] == "invalid":
            self._actor._hss_invalid_tool_feedback = evidence["feedback"]
        converted = copy(response)
        converted.content = json.dumps(normalized, ensure_ascii=False)
        return converted


def install_tool_decoding(stack, native):
    """Patch the step boundary only; planning, memory and paid SDK stay native."""
    original = native.ToolCallingAgent.step

    @wraps(original)
    def step(self, memory_step, *args, **kwargs):
        if not hasattr(self, "_hss_tool_decoding_events"):
            self._hss_tool_decoding_events = []
        self._hss_invalid_tool_feedback = None
        model = self.model
        self.model = _StepModel(model, self, memory_step)
        try:
            result = original(self, memory_step, *args, **kwargs)
            if self._hss_invalid_tool_feedback:
                memory_step.observations = self._hss_invalid_tool_feedback
            return result
        finally:
            self.model = model

    stack.enter_context(patch.object(native.ToolCallingAgent, "step", step))
