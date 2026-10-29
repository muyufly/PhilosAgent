"""Native text guidance accepts JSON memory without altering the provider."""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys

import pytest

from philoagent.hss_memory_format import guard_memory_response, normalize_memory_response


class Kind(Enum):
    TEXT = "text"
    API = "api"


@dataclass
class Item:
    id: str
    content: object
    metadata: dict = field(default_factory=dict)
    score: float = .7
    type: Kind = Kind.TEXT


@dataclass
class Response:
    memories: list
    total_count: int
    memory_type: str = "provider"
    request_id: str = "request"


@pytest.mark.parametrize("value", [["one", "二", {"source": "verbatim"}], {"facts": [1, True, None]},
                                  42, -3, 2.5, True, False, None])
def test_json_memory_is_copied_to_lossless_text_with_auditable_hashes(value):
    item = Item("memory", value, {"meaning": "preserve"})
    original = Response([item], 1)
    result, evidence = normalize_memory_response(original)
    assert result is not original and result.memories[0] is not item
    assert item.content is value
    assert json.loads(result.memories[0].content) == value
    assert result.total_count == original.total_count and result.request_id == original.request_id
    assert result.memories[0].metadata == item.metadata and result.memories[0].score == item.score
    assert evidence[0]["input_type"] == type(value).__name__
    assert len(evidence[0]["input_json_sha256"]) == 64
    assert len(evidence[0]["output_utf8_sha256"]) == 64


def test_text_and_api_tool_objects_preserve_identity_and_exact_text_bytes():
    text = "\r\n A source\tcontains 标点。\n"
    tool = object()
    response = Response([Item("text", text), Item("tool", tool, type=Kind.API)], 2)
    result, evidence = normalize_memory_response(response)
    assert result is response and not evidence
    assert result.memories[0].content == text and result.memories[0].content.encode() == text.encode()
    assert result.memories[1].content is tool


@pytest.mark.parametrize("value", [object(), {"not_json"}, b"bytes", ("tuple",), float("nan"), float("inf"), {1: "key"}])
def test_unsupported_text_memory_is_rejected_without_mutation(value):
    response = Response([Item("memory", value)], 1)
    with pytest.raises((TypeError, ValueError)):
        normalize_memory_response(response)
    assert response.memories[0].content is value


def test_cycle_is_rejected_without_inventing_a_text_fallback():
    value = []
    value.append(value)
    with pytest.raises(ValueError):
        normalize_memory_response(Response([Item("cycle", value)], 1))


def test_guard_records_interface_purpose_and_leaves_provider_object_untouched():
    response = Response([Item("memory", ["a", "b"])], 1)

    class Provider:
        def provide_memory(self, request):
            return response

        def _record(self, operation, **details):
            self.evidence = operation, details

    provider = Provider()
    wrapped = guard_memory_response(Provider.provide_memory)
    result = wrapped(provider, object())
    assert result.memories[0].content == '["a","b"]'
    assert response.memories[0].content == ["a", "b"]
    assert provider.evidence[0] == "memory_content_format"
    assert provider.evidence[1]["purpose"] == "native_text_interface_format"
    assert provider.evidence[1]["algorithm_changed"] is False


def test_real_native_guidance_consumes_structured_text_memory_without_join_failure(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = r'''
import json,sys,socket,os
from pathlib import Path
import openai,dotenv
def forbidden(*args,**kwargs):raise AssertionError('Memory formatting test attempted network or model access')
openai.OpenAI.__init__=forbidden
socket.socket.connect=forbidden
dotenv.load_dotenv=lambda *args,**kwargs:False
dotenv.dotenv_values=lambda *args,**kwargs:{}
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
sys.path.insert(0,str(Path.cwd()/'vendor/MemEvolve/Flash-Searcher-main'))
from FlashOAgents.agents import ToolCallingAgent
from EvolveLab.memory_types import MemoryItem,MemoryItemType,MemoryResponse,MemoryType,MemoryStatus
from philoagent.hss_memory_format import guard_memory_response
response=MemoryResponse([MemoryItem('digest',['one','二','three','four','five'],{},type=MemoryItemType.TEXT)],MemoryType.LIGHTWEIGHT_MEMORY,1)
class Provider:
 def provide_memory(self,request):return response
 def _record(self,*args,**kwargs):pass
provider=Provider()
provider.provide_memory=guard_memory_response(Provider.provide_memory).__get__(provider,Provider)
agent=object.__new__(ToolCallingAgent)
agent.memory_provider=provider
agent.task='Original task'
agent.tools={}
agent._format_current_context=lambda:''
text=agent._get_memory_guidance(MemoryStatus.IN,step_number=2)
assert json.loads(text)==['one','二','three','four','five']
assert response.memories[0].content==['one','二','three','four','five']
print('NATIVE_MEMORY_JSON_INTERFACE_OK')
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=project,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr)[-7000:]
    assert "NATIVE_MEMORY_JSON_INTERFACE_OK" in result.stdout
