"""Native tool decoding failures must remain recoverable within one actor."""

from contextlib import ExitStack
from types import SimpleNamespace
import json

import pytest

from philoagent.hss_tool_calls import (InvalidToolResponse, decode_tool_response,
                                      install_tool_decoding, recover_final_answer)


def message(content=None, tools=None):
    return SimpleNamespace(content=content, tool_calls=tools)


def test_plan_string_is_format_feedback_instead_of_a_tool_object():
    normalized, evidence = decode_tool_response(message('{"tools":["PLAN"]}'), {"web_search", "final_answer"})
    assert normalized["tools"] == []
    assert evidence["status"] == "invalid"
    assert "original_content" in evidence
    assert "tools" in evidence["feedback"]


def test_dsml_analysis_then_parallel_invocations_preserves_exact_arguments():
    content = '''[Analysis]
The text is not a final answer.
[Tools]
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="web_search">
<｜｜DSML｜｜ parameter name="arguments" string="false">{"query":"中国 2024 OSI 0.9258"}</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="crawl_page">
<｜｜DSML｜｜ parameter name="url" string="true">https://example.org/report?a=1&amp;b=2</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="query" string="true">Evidence &amp; limitations</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''
    normalized, evidence = decode_tool_response(message(content), {"web_search", "crawl_page"})
    assert normalized["tools"] == [
        {"name": "web_search", "arguments": {"query": "中国 2024 OSI 0.9258"}},
        {"name": "crawl_page", "arguments": {"url": "https://example.org/report?a=1&b=2", "query": "Evidence & limitations"}},
    ]
    assert evidence["status"] == "normalized"
    assert evidence["format"] == "dsml"


def test_structured_openai_calls_are_decoded_without_content():
    tool = SimpleNamespace(id="call1", function=SimpleNamespace(name="final_answer", arguments='{"answer":"完整作品"}'))
    normalized, evidence = decode_tool_response(message(tools=[tool]), {"final_answer"})
    assert normalized["tools"] == [{"name": "final_answer", "arguments": {"answer": "完整作品"}, "id": "call1"}]
    assert evidence["format"] == "openai_tool_calls"


@pytest.mark.parametrize("nested", [False, True])
def test_unescaped_quotes_preserve_entire_final_report_instead_of_repairing_a_prefix(nested):
    report = '白皮书《中国"互联网+政务服务"发展现状》\n第一章：完整正文。\n第六章：建议。\n附录：全部来源。'
    body = json.dumps(report, ensure_ascii=False)[1:-1].replace('\\"', '"')
    argument = '{"answer":"' + body + '"}' if nested else '"' + body + '"'
    raw = '{"think":"完成。","tools":[{"name":"final_answer","arguments":' + argument + '}]}'
    normalized, evidence = decode_tool_response(message(raw), {"final_answer"})
    assert normalized["tools"] == [{"name": "final_answer", "arguments": {"answer": report}}]
    assert evidence["status"] == "normalized"
    assert evidence["final_answer_recovery"]["lossless"]


def test_lossless_recovery_decodes_escapes_once_and_keeps_embedded_json_and_final_suffix():
    report = ('标题"引文"\n路径 C:\\data\\file；字面量 \\n；emoji 😀。\n'
              '示例 {"report":"仅为文内例子","quote":"保持全部内容"}\n'
              '第六章：完整建议。\n附录：最后一条来源。')
    body = json.dumps(report, ensure_ascii=False)[1:-1].replace('\\"', '"')
    raw = '{"tools":[{"name":"final_answer","arguments":"' + body + '"}]}'
    normalized, evidence = recover_final_answer(message(raw), {"final_answer"})
    answer = normalized["tools"][0]["arguments"]["answer"]
    assert answer == report
    assert evidence["source_body_chars"] == len(body)
    start, end = evidence["source_body_span"]
    assert raw[start:end] == body
    assert evidence["decoded_answer_chars"] == len(report)


def test_long_final_report_keeps_all_six_chapters_after_early_unescaped_quotes():
    report = ('中国"互联网+政务服务"白皮书\n'
              + '\n'.join(f'第{chapter}章：' + '完整证据与论证。' * 250 for chapter in range(1, 7))
              + '\n附录：最后一条来源保留。')
    raw_body = json.dumps(report, ensure_ascii=False)[1:-1].replace('\\"', '"')
    raw = '{"tools":[{"name":"final_answer","arguments":"' + raw_body + '"}]}'
    normalized, evidence = decode_tool_response(message(raw), {"final_answer"})
    assert normalized["tools"][0]["arguments"]["answer"] == report
    assert len(report) > 10000
    assert evidence["final_answer_recovery"]["decoded_answer_chars"] == len(report)


@pytest.mark.parametrize("raw", [
    '{"tools":[{"name":"final_answer","arguments":"标题"引文"\\n尚未结束',
    '{"tools":[{"name":"final_answer","arguments":"标题"引文""}]',
    '{"tools":[{"name":"final_answer","arguments":"标题"引文"\\q非法转义"}]}',
    '{"tools":[],"tools":[{"name":"final_answer","arguments":{"answer":"重复字段"}}]}',
    '{"tools":[{"name":"final_answer","arguments":{"answer":"一","answer":"二"}}]}',
    '{"tools":[{"name":"final_answer","arguments":"标题"引文""},'
    '{"name":"final_answer","arguments":"另一个作品"}]}',
    '{tools:[{name:"final_answer",arguments:{answer:"外壳也不明确"}}]}',
])
def test_ambiguous_or_unfinished_final_answer_never_uses_permissive_prefix_repair(raw):
    normalized, evidence = decode_tool_response(message(raw), {"final_answer"})
    assert normalized["tools"] == []
    assert evidence["status"] == "invalid"
    assert evidence["original_content"] == raw


def test_lossless_recovery_requires_declared_final_answer():
    raw = '{"tools":[{"name":"final_answer","arguments":"完整文本"}]}'
    with pytest.raises(InvalidToolResponse):
        recover_final_answer(message(raw), {"web_search"})


@pytest.mark.parametrize("wrapper", ["fence", "list", "multi"])
def test_strict_valid_final_answers_keep_existing_envelope_support(wrapper):
    action = {"name": "final_answer", "arguments": {"answer": '完整"作品"及附录。'}}
    value = {"tools": [action]}
    if wrapper == "list":
        value = [action]
    elif wrapper == "multi":
        value["tools"].insert(0, {"name": "web_search", "arguments": {"query": "evidence"}})
    raw = json.dumps(value, ensure_ascii=False)
    if wrapper == "fence":
        raw = "```json\n" + raw + "\n```"
    normalized, evidence = decode_tool_response(message(raw), {"final_answer", "web_search"})
    assert normalized["tools"][-1] == action
    assert evidence["status"] != "invalid"


def test_harmless_nonfinal_action_punctuation_still_uses_native_json_compatibility():
    raw = '{tools:[{name:"web_search",arguments:{query:"完整查询"}}]}'
    normalized, evidence = decode_tool_response(message(raw), {"web_search", "final_answer"})
    assert normalized["tools"] == [{"name": "web_search", "arguments": {"query": "完整查询"}}]
    assert evidence["status"] == "normalized"


@pytest.mark.parametrize("content", [
    '{"tools":[{"name":"undeclared","arguments":{}}]}',
    '{"tools":[{"name":"web_search","arguments":"query"}]}',
    '{"tools":["Analysis] arbitrary content"]}',
    '<｜｜DSML｜｜ invoke name="web_search"><｜｜DSML｜｜ parameter name="arguments" string="false">{"query":"truncated"}',
    'A substantive report that never explicitly calls final_answer.',
])
def test_malformed_or_undeclared_actions_never_become_executable_tools_or_answers(content):
    normalized, evidence = decode_tool_response(message(content), {"web_search", "final_answer"})
    assert normalized["tools"] == []
    assert evidence["status"] == "invalid"


def test_scoped_step_adapter_continues_same_agent_and_records_paid_original():
    class Actor:
        def step(self, memory_step):
            response = self.model([{"role": "user", "content": "step"}])
            actions = json.loads(response.content)["tools"]
            memory_step.tool_calls = []
            for action in actions:
                if action["name"] == "final_answer":
                    return action["arguments"]["answer"]
            return None

    responses = iter([message('{"tools":["PLAN"]}'),
                      message('{"tools":[{"name":"final_answer","arguments":{"answer":"Kept work"}}]}')])
    original_model = lambda messages: next(responses)
    actor = Actor()
    actor.model = original_model
    actor.tools = {"final_answer": object()}
    actor.step_number = 1
    events = []
    actor._hss_decoding_observer = events.append
    first = SimpleNamespace(observations=None)
    second = SimpleNamespace(observations=None)
    native = SimpleNamespace(ToolCallingAgent=Actor)
    with ExitStack() as stack:
        install_tool_decoding(stack, native)
        assert actor.step(first) is None
        assert "tools" in first.observations
        actor.step_number += 1
        assert actor.step(second) == "Kept work"
    assert actor.model is original_model
    assert len(events) == 2
    assert events[0]["original_content"] == '{"tools":["PLAN"]}'
    assert events[0]["step_number"] == 1
    assert events[1]["status"] == "valid"
