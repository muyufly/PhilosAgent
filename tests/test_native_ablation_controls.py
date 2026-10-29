"""Native ablation control and resume regressions; no secrets or network access."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_evolution_integration import AUTO, CREATOR, UPSTREAM, load_upstream
from philoagent.orientation import bootstrap, load

VALIDATOR = load_upstream("MemEvolve/phases/phase_validator.py", "MemEvolve.phases.phase_validator")


class NativeAblationControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("PHILO_LINEAGE_ROOT", None)
        silence = contextlib.redirect_stdout(io.StringIO())
        silence.__enter__()
        self.addCleanup(silence.__exit__, None, None, None)

    def alias_fixture(self):
        package = self.root / "EvolveLab"
        (package / "providers").mkdir(parents=True)
        (package / "memory_types.py").write_text(
            'from enum import Enum\nclass MemoryType(Enum):\n    BASE = "base"\n'
            '    # add new memory type upside this line(Enum)\n'
            'PROVIDER_MAPPING = {\n    MemoryType.BASE: ("BaseProvider", "base_provider"),\n'
            '# add new memory type upside this line(PROVIDER_MAPPING)\n}\n', encoding="utf-8")
        (package / "config.py").write_text(
            "providers = {\n# add new memory type upside this line\n}\n", encoding="utf-8")
        source = package / "providers" / "base_provider.py"
        # Mixed line endings catch accidental text reserialization on any host.
        source.write_bytes(b"class BaseProvider:\r\n    pass\n")
        bootstrap(self.root / "lineage", "base", ["Original method."])
        os.environ["PHILO_LINEAGE_ROOT"] = str(self.root / "lineage")
        config = {
            "provider_code": {"class_name": "BaseProvider", "module_name": "base_provider",
                              "code": source.read_text(encoding="utf-8")},
            "memory_type_info": {"enum_name": "CHILD", "enum_value": "child"},
            "config_updates": {"storage_dir": "./storage/child", "enable_longterm_provision": True},
            "_orientation_parent": {"lineage_root": str(self.root / "lineage"),
                                    "snapshot": load(self.root / "lineage", "base").to_dict()},
            "orientation_items": ["Revised method."],
            "_philo_freeze_code": {"module_name": "base_provider", "class_name": "BaseProvider",
                                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        }
        return source, config

    def test_fixed_code_alias_registers_and_forks_phi_without_touching_source_bytes(self):
        source, config = self.alias_fixture()
        before = source.read_bytes()
        result = CREATOR.MemorySystemCreator.create_memory_system(config, self.root)
        self.assertTrue(result["success"], result)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(load(self.root / "lineage", "child").items, ("Revised method.",))
        mapping = (self.root / "EvolveLab" / "memory_types.py").read_text(encoding="utf-8")
        self.assertIn('MemoryType.CHILD: ("BaseProvider", "base_provider")', mapping)
        self.assertTrue(result["results"]["provider_source_frozen"])

    def test_fixed_code_alias_rejects_hash_or_identity_changes_before_registration(self):
        for field, value in (("source_sha256", "0" * 64), ("module_name", "other"),
                             ("class_name", "OtherProvider")):
            with self.subTest(field=field):
                source, config = self.alias_fixture_reusable()
                config["_philo_freeze_code"][field] = value
                before = source.read_bytes()
                result = CREATOR.MemorySystemCreator.create_memory_system(config, self.root)
                self.assertFalse(result["success"])
                self.assertEqual(source.read_bytes(), before)
                self.assertFalse((self.root / "lineage" / "child").exists())
                self.assertNotIn('CHILD = "child"', (self.root / "EvolveLab" / "memory_types.py").read_text())

    def alias_fixture_reusable(self):
        if not (self.root / "EvolveLab").exists():
            return self.alias_fixture()
        source = self.root / "EvolveLab" / "providers" / "base_provider.py"
        config = {
            "provider_code": {"class_name": "BaseProvider", "module_name": "base_provider",
                              "code": source.read_text(encoding="utf-8")},
            "memory_type_info": {"enum_name": "CHILD", "enum_value": "child"},
            "config_updates": {"storage_dir": "./storage/child"},
            "_orientation_parent": {"lineage_root": str(self.root / "lineage"),
                                    "snapshot": load(self.root / "lineage", "base").to_dict()},
            "_philo_freeze_code": {"module_name": "base_provider", "class_name": "BaseProvider",
                                   "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        }
        return source, config

    def auto(self):
        auto = AUTO.AutoEvolver(
            analysis_model_id="offline", gen_model_id="offline", work_root=str(self.root / "run"),
            dataset_name="fixture", datasets_config={"fixture": str(self.root / "data.jsonl")},
            run_provider=lambda *args: args[-1], default_provider="parent", num_systems=1,
            task_batch_x=2, top_t=1, extra_sample_y=1, clear_storage_per_round=False)
        auto._run_memory_evolver = lambda *args: {"validated_systems": ["child"]}
        auto._evaluate_providers_parallel = lambda **kwargs: {
            provider: {"success": True, "summary": {"accuracy": 1, "tokens": {"total_tokens": 1}}}
            for provider in kwargs["providers"]}
        return auto

    def test_resume_uses_saved_finalist_tasks_without_resampling(self):
        auto = self.auto()
        expected = [1, 2, 3]
        auto._save_checkpoint(0, {"parent_provider": "parent", "step_completed": 3,
                                 "batch_indices": [0, 1], "finalist_tasks": expected})
        calls = []
        original = auto._evaluate_providers_parallel

        def evaluate(**kwargs):
            calls.append(kwargs["tasks"])
            return original(**kwargs)

        auto._evaluate_providers_parallel = evaluate
        with patch.object(AUTO.random, "sample", return_value=[0]) as sample:
            result = auto.run()
        sample.assert_not_called()
        self.assertEqual(calls[-1], expected)
        self.assertEqual(result["history"][0]["finalist_tasks"], expected)

    def test_all_failed_finalists_raise_explicit_failure(self):
        auto = self.auto()
        original = auto._evaluate_providers_parallel

        def evaluate(**kwargs):
            if kwargs["round_dir"].name == "eval_finalists":
                return {provider: {"success": False} for provider in kwargs["providers"]}
            return original(**kwargs)

        auto._evaluate_providers_parallel = evaluate
        with self.assertRaisesRegex(RuntimeError, "No successful finalist"):
            auto.run()

    def test_resumed_history_file_preserves_all_completed_round_summaries(self):
        auto = self.auto()
        auto.run()
        result = auto.run()
        stored = json.loads(Path(result["history_path"]).read_text(encoding="utf-8"))
        self.assertEqual([row["round"] for row in stored], [0, 1])
        self.assertEqual(result["rounds"], 1)

    def validation(self, *, require_orientation=False, invalid_response=False, ingestion_success=True):
        lineage = self.root / "validation_lineage"
        bootstrap(lineage, "lightweight_memory", ["Apply this method in validation."])
        os.environ["PHILO_LINEAGE_ROOT"] = str(lineage)
        isolated = self.root / "isolated"
        package = isolated / "EvolveLab"
        (package / "providers").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "providers" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(UPSTREAM / "EvolveLab" / "memory_types.py", package / "memory_types.py")
        (package / "config.py").write_text(
            'def get_memory_config(memory_type):\n    return {"storage_dir": "./storage/lightweight_memory"}\n',
            encoding="utf-8")
        (package / "providers" / "lightweight_memory_provider.py").write_text(
            'from EvolveLab.memory_types import MemoryResponse, MemoryType\n'
            'class LightweightMemoryProvider:\n'
            '    def __init__(self, config):\n        self.config = config\n'
            '    def initialize(self):\n'
            f'        return not {require_orientation!r} or self.config.get("philo_orientation_items") == ["Apply this method in validation."]\n'
            '    def provide_memory(self, request):\n'
            f'        return "invalid response" if {invalid_response!r} else MemoryResponse([], MemoryType.LIGHTWEIGHT_MEMORY, 0)\n'
            '    def take_in_memory(self, trajectory):\n'
            f'        return ({ingestion_success!r}, "offline ingestion result")\n', encoding="utf-8")
        validator = VALIDATOR.PhaseValidator(self.root / "validation", "unused.json", cleanup_temp=False)
        validator.isolated_env_dir = isolated
        flash = types.ModuleType("FlashOAgents")
        flash.OpenAIServerModel = lambda **kwargs: object()
        base = types.ModuleType("base_agent")
        base.SearchAgent = object
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"FlashOAgents": flash, "base_agent": base, "dotenv": dotenv}):
            result = validator._run_fast_validation("lightweight_memory", setup_env=False)
        return result

    def test_validator_uses_native_orientation_loader(self):
        result = self.validation(require_orientation=True)
        self.assertEqual(result["verdict"], "passed", result)
        self.assertEqual(result["tests"]["provide_memory"]["memories_count"], 1)

    def test_validator_rejects_wrong_memory_response_type(self):
        result = self.validation(invalid_response=True)
        self.assertEqual(result["verdict"], "failed", result)
        self.assertEqual(result["tests"]["provide_memory"]["status"], "failed")

    def test_validator_rejects_explicit_false_ingestion(self):
        result = self.validation(ingestion_success=False)
        self.assertEqual(result["verdict"], "failed", result)
        self.assertEqual(result["tests"]["take_in_memory"]["status"], "failed")

    def test_experimental_validator_disables_automatic_code_fixes(self):
        os.environ["PHILO_LINEAGE_ROOT"] = "offline-experiment"
        validator = VALIDATOR.PhaseValidator(self.root / "validation", "unused.json", enable_auto_fix=True)
        self.assertFalse(validator.enable_auto_fix)


if __name__ == "__main__":
    unittest.main()
