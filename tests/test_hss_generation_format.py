"""Registration repairs parse saved text only; generated Python never executes."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from philoagent.hss_generation_format import recover_missing_registration


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_parser(monkeypatch):
    config = types.ModuleType("format_fixture.config")
    config.CREATIVITY_INDEX, config.CREATIVITY_TEMP_BASE, config.CREATIVITY_TEMP_SPAN = .5, .5, .5
    monkeypatch.setitem(sys.modules, config.__name__, config)
    source = ROOT / "vendor/MemEvolve/Flash-Searcher-main/MemEvolve/phases/phase_generator.py"
    spec = importlib.util.spec_from_file_location("format_fixture.phases.phase_generator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhaseGenerator._parse_system_config.__get__(object())


@pytest.fixture
def response():
    return (
        "## Provider Information\n**Class Name**: GradusMemoryProvider\n"
        "**Module Name**: gradus_memory_provider\n## Provider Code\n```python\n"
        "from EvolveLab.base_memory import BaseMemoryProvider\n"
        "from EvolveLab.memory_types import MemoryType\n"
        "raise RuntimeError('Generated source must never execute in a format repair')\n"
        "class GradusMemoryProvider(BaseMemoryProvider):\n"
        "    def __init__(self, config):\n"
        "        self.config = config\n"
        "        self.storage_dir = self.config.get('storage_dir', './storage/gradus_memory')\n"
        "        self.memory_type = MemoryType.GRADUS_MEMORY\n```\n")


def test_real_native_missing_registration_red_seam_and_lossless_recovery(native_parser, response):
    assert native_parser(response) is None
    original = response
    config = recover_missing_registration(response, native_parser)
    assert config["memory_type_info"] == {"enum_name": "GRADUS_MEMORY", "enum_value": "gradus_memory"}
    assert config["config_updates"] == {}
    expected_code = response.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
    assert config["provider_code"] == {"class_name": "GradusMemoryProvider",
        "module_name": "gradus_memory_provider", "code": expected_code}
    assert response == original
    audit = config["_generation_format_repair"]
    assert audit["original_response_sha256"] == hashlib.sha256(response.encode()).hexdigest()
    assert audit["provider_code_sha256"] == hashlib.sha256(expected_code.encode()).hexdigest()
    assert audit["added_fields"] == {"Enum Name": "GRADUS_MEMORY", "Enum Value": "gradus_memory"}


def test_explicit_configuration_and_phi_are_preserved(native_parser, response):
    response += ('## Configuration\n**top_k**: 17\n'
                 '**orientation_items**: ["Keep this paid principle unchanged."]\n')
    config = recover_missing_registration(response, native_parser)
    assert config["config_updates"] == {"top_k": "17"}
    assert config["orientation_items"] == ["Keep this paid principle unchanged."]


def test_matching_partial_registration_is_completed(native_parser, response):
    response += "**Enum Name**: GRADUS_MEMORY\n"
    config = recover_missing_registration(response, native_parser)
    assert config["_generation_format_repair"]["added_fields"] == {"Enum Value": "gradus_memory"}


@pytest.mark.parametrize("change", [
    lambda text: text.replace("self.memory_type = MemoryType.GRADUS_MEMORY", "self.memory_type = (MemoryType.GRADUS_MEMORY, MemoryType.OTHER_MEMORY)"),
    lambda text: text.replace("self.memory_type = MemoryType.GRADUS_MEMORY", "self.other_path = './storage/other_memory'\n        self.memory_type = MemoryType.GRADUS_MEMORY"),
    lambda text: text.replace("**Module Name**: gradus_memory_provider", "**Module Name**: other_provider"),
    lambda text: text.replace("class GradusMemoryProvider", "class OtherProvider"),
    lambda text: text + "**Enum Name**: OTHER_MEMORY\n",
    lambda text: text + "**Enum Value**: other_memory\n",
    lambda text: text.replace("class GradusMemoryProvider(BaseMemoryProvider):", "class GradusMemoryProvider(BaseMemoryProvider)"),
    lambda text: text.replace("**Class Name**: GradusMemoryProvider\n", ""),
    lambda text: text + "**Module Name**: gradus_memory_provider\n",
    lambda text: text.replace("raise RuntimeError", "MemoryType = object()\nraise RuntimeError"),
    lambda text: text.replace("raise RuntimeError", "class MemoryType: pass\nraise RuntimeError"),
    lambda text: text.replace("self.memory_type = MemoryType.GRADUS_MEMORY", "self.other = getattr(MemoryType, name)\n        self.memory_type = MemoryType.GRADUS_MEMORY"),
    lambda text: text.replace("self.memory_type = MemoryType.GRADUS_MEMORY", "MemoryType.GRADUS_MEMORY = None"),
    lambda text: text.replace("import MemoryType", "import MemoryType as OtherType"),
    lambda text: text.replace("self.config.get('storage_dir', './storage/gradus_memory')", "self.config.get('another_path', './storage/gradus_memory')"),
    lambda text: text + "```python\nclass AnotherProvider: pass\n```\n",
])
def test_ambiguous_conflicting_or_incomplete_evidence_is_rejected(native_parser, response, change):
    with pytest.raises(ValueError):
        recover_missing_registration(change(response), native_parser)


def test_complete_registration_does_not_trigger_recovery(native_parser, response):
    response += "**Enum Name**: GRADUS_MEMORY\n**Enum Value**: gradus_memory\n"
    original = deepcopy(native_parser(response))
    assert recover_missing_registration(response, native_parser) is None
    assert native_parser(response) == original


@pytest.mark.parametrize("policy", ["restricted", "native_local"])
def test_policy_parser_hook_is_scoped_reversible_and_idempotent(tmp_path, monkeypatch, native_parser, response, policy):
    from philoagent.ablation_policy import install_native_policy
    workspace = tmp_path / "workspace"
    native_path = workspace / "MemEvolve/phases/phase_generator.py"
    native_path.parent.mkdir(parents=True)
    native_path.write_text("# native controller fixture\n", encoding="utf-8")

    class Generator:
        _parse_system_config = native_parser.__func__

        def _build_generation_prompt(self, *args, **kwargs):
            return "prompt"

        def _generate_single_system(self, *args, **kwargs):
            return None

    module = types.ModuleType("MemEvolve.phases.phase_generator")
    module.__file__, module.PhaseGenerator = str(native_path), Generator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = Generator._parse_system_config
    job = {"job_id": "format-test", "code_evolve": True, "orientation_evolve": True}
    restore = install_native_policy(tmp_path, job, generated_code_policy=policy)
    try:
        assert install_native_policy(tmp_path, job, generated_code_policy=policy) is restore
        parsed = Generator()._parse_system_config(response)
        if policy == "restricted":
            assert parsed is None and Generator._parse_system_config is original
        else:
            assert parsed["memory_type_info"]["enum_value"] == "gradus_memory"
            assert parsed["_raw_llm_output"] == response
            complete = response + "**Enum Name**: GRADUS_MEMORY\n**Enum Value**: gradus_memory\n"
            assert Generator()._parse_system_config(complete) == original(Generator(), complete)
    finally:
        restore()
    assert Generator._parse_system_config is original
    assert not hasattr(Generator, "_philo_ablation_policy")
