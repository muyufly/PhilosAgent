"""Policy tests use synthetic generator output and never contact a model."""

import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from philoagent.ablation_policy import install_native_policy
from philoagent.orientation import bootstrap, load


PROJECT = Path(__file__).resolve().parents[1]


class AblationPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "run"
        self.workspace = self.run / "workspace"
        package = self.workspace / "EvolveLab"
        (package / "providers").mkdir(parents=True)
        self.source = package / "providers/lightweight_memory_provider.py"
        self.source.write_bytes(b'class LightweightMemoryProvider:\r\n    pass\r\n')
        (package / "memory_types.py").write_text(
            'from enum import Enum\nclass MemoryType(Enum):\n'
            '    LIGHTWEIGHT_MEMORY = "lightweight_memory"\n'
            '    # add new memory type upside this line(Enum)\n'
            'PROVIDER_MAPPING = {\n'
            '    MemoryType.LIGHTWEIGHT_MEMORY: ("LightweightMemoryProvider", "lightweight_memory_provider"),\n'
            '    # add new memory type upside this line(PROVIDER_MAPPING)\n}\n', encoding="utf-8")
        self.config_path = package / "config.py"
        self.config_path.write_text(
            'DEFAULT_CONFIG = {"providers": {\n'
            '    MemoryType.LIGHTWEIGHT_MEMORY: {"model": None, '
            '"storage_dir": "./storage/lightweight_memory", "top_k": 3, '
            '"nested_settings": {"threshold": 0.2}, "enable_longterm_provision": False},\n'
            '    # add new memory type upside this line\n}}\n', encoding="utf-8")
        bootstrap(self.run / "lineage", "lightweight_memory", ["Current parent criterion."])
        self.snapshot = load(self.run / "lineage", "lightweight_memory")
        native_file = self.workspace / "MemEvolve/phases/phase_generator.py"
        native_file.parent.mkdir(parents=True)
        native_file.write_text("# synthetic generator\n", encoding="utf-8")

        class FakeGenerator:
            def _build_generation_prompt(self, *args, **kwargs):
                return "Native current-parent prompt."

            def _generate_single_system(self, *args, **kwargs):
                return deepcopy(self.proposal)

        self.generator_class = FakeGenerator
        self.generator = FakeGenerator()
        self.generator.default_provider = "lightweight_memory"
        self.generator._orientation_snapshot = self.snapshot
        self.generator.work_dir = self.run / "evolution/round_00"
        self.generator.proposal = self.proposal()
        module = types.ModuleType("MemEvolve.phases.phase_generator")
        module.__file__ = str(native_file)
        module.PhaseGenerator = FakeGenerator
        patcher = patch.dict(sys.modules, {module.__name__: module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.job = {"job_id": "current-p0-phi0-r00", "code_evolve": False, "orientation_evolve": False}

    @staticmethod
    def proposal():
        return {
            "provider_code": {"class_name": "NewProvider", "module_name": "new_provider",
                              "code": "import json\nimport os\nclass NewProvider:\n    pass\n"},
            "memory_type_info": {"enum_name": "NEW_PROVIDER", "enum_value": "new_memory"},
            "config_updates": {"top_k": "99", "storage_dir": "'./storage/new_memory'"},
            "orientation_items": ["Model proposed a revised criterion."],
        }

    def install(self, *, code=False, orientation=False, **kwargs):
        job = dict(self.job, code_evolve=code, orientation_evolve=orientation)
        restore = install_native_policy(self.run, job, **kwargs)
        self.addCleanup(restore)
        return restore

    def project(self):
        return self.generator._generate_single_system("unused prompt", 0.2)

    def test_frozen_code_config_and_orientation_with_alias_creation(self):
        before = self.source.read_bytes()
        before_hash = hashlib.sha256(before).hexdigest()
        self.install()
        projected = self.project()
        self.assertEqual(projected["provider_code"]["module_name"], "lightweight_memory_provider")
        self.assertEqual(projected["_philo_freeze_code"]["source_sha256"], before_hash)
        config = projected["config_updates"]
        self.assertEqual(config["top_k"], 3)
        self.assertEqual(config["nested_settings"], {"threshold": 0.2})
        self.assertTrue(config["enable_longterm_provision"])
        self.assertNotIn("model", config)
        alias = projected["memory_type_info"]["enum_value"]
        self.assertLess(len(alias), 100)
        self.assertEqual(config["storage_dir"], f"./storage/{alias}")
        self.assertEqual(projected["orientation_items"], list(self.snapshot.items))
        creator_file = PROJECT / "vendor/MemEvolve/Flash-Searcher-main/MemEvolve/phases/memory_creator.py"
        spec = importlib.util.spec_from_file_location("policy_fixture_creator", creator_file)
        creator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(creator)
        with patch.dict("os.environ", {"PHILO_LINEAGE_ROOT": str(self.run / "lineage")}), contextlib.redirect_stdout(io.StringIO()):
            result = creator.MemorySystemCreator.create_memory_system(projected, str(self.workspace))
        self.assertTrue(result["success"], result)
        self.assertTrue(result["results"]["provider_source_frozen"])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(load(self.run / "lineage", alias).items, self.snapshot.items)
        self.assertIn(f'MemoryType.{alias.upper()}: ("LightweightMemoryProvider", "lightweight_memory_provider")',
                      (self.workspace / "EvolveLab/memory_types.py").read_text(encoding="utf-8"))
        event = json.loads((self.run / "evidence/factor_projection.jsonl").read_text())
        self.assertEqual(event["parent_code_sha256"], event["child_code_sha256"])
        self.assertEqual(event["parent_effective_config_sha256"], event["child_effective_config_sha256"])
        self.assertEqual(event["parent_items_sha256"], event["child_items_sha256"])
        self.assertTrue(event["orientation_proposal_overridden"])
        # On the next round, disk AST finds the newly registered alias even if
        # Python's imported native registries still contain only the baseline.
        self.generator.default_provider = alias
        self.generator._orientation_snapshot = load(self.run / "lineage", alias)
        self.generator.work_dir = self.run / "evolution/round_01"
        next_config = self.project()["config_updates"]
        self.assertTrue(next_config["enable_longterm_provision"])
        self.assertEqual(next_config["top_k"], 3)
        self.assertNotEqual(next_config["storage_dir"], config["storage_dir"])

    def test_orientation_evolving_preserves_missing_empty_and_revised_proposals(self):
        self.install(orientation=True)
        for proposal in (None, [], ["New evidence-scoped criterion."]):
            with self.subTest(proposal=proposal):
                if proposal is None:
                    self.generator.proposal.pop("orientation_items", None)
                else:
                    self.generator.proposal["orientation_items"] = proposal
                projected = self.project()
                if proposal is None:
                    self.assertNotIn("orientation_items", projected)
                else:
                    self.assertEqual(projected["orientation_items"], proposal)

    def test_code_evolving_keeps_source_and_projects_frozen_orientation(self):
        self.install(code=True)
        projected = self.project()
        self.assertEqual(projected["provider_code"], self.generator.proposal["provider_code"])
        self.assertNotIn("_philo_freeze_code", projected)
        self.assertEqual(projected["config_updates"]["top_k"], 99)
        self.assertEqual(projected["orientation_items"], list(self.snapshot.items))

    def test_two_pending_candidates_with_same_namespace_are_disambiguated_before_creation(self):
        self.install(code=True, generated_code_policy="native_local")
        first = self.project()
        self.generator.work_dir.mkdir(parents=True)
        path = self.generator.work_dir / "generated_system_1.json"
        path.write_text(json.dumps(first), encoding="utf-8")
        original = deepcopy(self.generator.proposal)
        second = self.project()
        self.assertNotEqual(second["memory_type_info"]["enum_value"], first["memory_type_info"]["enum_value"])
        self.assertNotEqual(second["provider_code"]["class_name"], first["provider_code"]["class_name"])
        self.assertNotEqual(second["provider_code"]["module_name"], first["provider_code"]["module_name"])
        self.assertNotEqual(second["config_updates"]["storage_dir"], first["config_updates"]["storage_dir"])
        self.assertEqual(self.generator.proposal, original)
        self.assertEqual(json.loads(path.read_text()), first)

    def test_native_docstring_bold_labels_do_not_become_configuration_fields(self):
        self.install(code=True, generated_code_policy="native_local", compact_provider=False)
        # Minimized from the paid WarrantMemoryProvider system-2 response. The
        # native parser scans the whole response, including provider docstrings.
        self.generator.proposal["config_updates"].update({
            "Constraint seeding": "quotas are pinned", "transfer_criterion": "an observable signal"})
        self.generator.proposal["_raw_llm_output"] = (
            '**Class Name**: NewProvider\n**Module Name**: new_provider\n'
            '```python\nclass NewProvider:\n    """\n'
            '**Constraint seeding**: quotas are pinned\n'
            '**transfer_criterion**: an observable signal\n'
            '    """\n```\n## Configuration\n'
            '**top_k**: 99\n**storage_dir**: ./storage/new_memory\n'
            '## Memory Type\n**Enum Name**: NEW_PROVIDER\n**Enum Value**: new_memory\n')
        projected = self.project()
        self.assertEqual(projected["config_updates"], {"top_k": 99, "storage_dir": "./storage/new_memory"})
        self.assertEqual(projected["_config_extraction"]["discarded_native_keys"],
                         ["Constraint seeding", "transfer_criterion"])
        self.assertEqual(projected["_config_extraction"]["method"], "explicit_configuration_section")

    def test_native_config_without_section_heading_ignores_code_fenced_labels(self):
        self.install(code=True, generated_code_policy="native_local")
        self.generator.proposal["config_updates"]["transfer_criterion"] = "not a setting"
        self.generator.proposal["_raw_llm_output"] = (
            '**Class Name**: NewProvider\n**Module Name**: new_provider\n'
            '```python\nclass NewProvider:\n    """\n'
            '## Configuration\n**transfer_criterion**: not a setting\n    """\n```\n'
            '**top_k**: 99\n**storage_dir**: ./storage/new_memory\n'
            '**Enum Name**: NEW_PROVIDER\n**Enum Value**: new_memory\n')
        projected = self.project()
        self.assertEqual(projected["config_updates"], {"top_k": 99, "storage_dir": "./storage/new_memory"})
        self.assertEqual(projected["_config_extraction"]["method"], "outside_code_fences")

    def test_declared_config_values_preserve_normal_literal_parsing_and_boundaries(self):
        self.install(code=True, generated_code_policy="native_local")
        self.generator.proposal["_raw_llm_output"] = (
            '## Configuration\n**storage_dir**: ./storage/new_memory\n**top_k**: 99\n'
            '**stage_boundaries**: [5, 12]\n**enabled**: true\n**alpha**: 0.25\n'
            '## Explanation\n**transfer_criterion**: not a setting\n')
        projected = self.project()
        self.assertEqual(projected["config_updates"], {"storage_dir": "./storage/new_memory",
            "top_k": 99, "stage_boundaries": [5, 12], "enabled": True, "alpha": .25})

    def test_actual_configuration_still_rejects_invalid_names_and_storage_escape(self):
        self.install(code=True, generated_code_policy="native_local")
        for fields, error in (("**bad key**: 5", "identifiers"),
                              ("**storage_dir**: ../outside", "escapes workspace/storage"),
                              ("**top_k**: 1\n**top_k**: 99", "conflicting values")):
            with self.subTest(fields=fields):
                self.generator.proposal["_raw_llm_output"] = "## Configuration\n" + fields + "\n"
                with self.assertRaisesRegex(ValueError, error):
                    self.project()

    def test_native_local_policy_accepts_real_candidate_algorithm_and_storage_patterns(self):
        self.install(code=True, generated_code_policy="native_local", compact_provider=False)
        # Minimal versions of the three rejected formal-run candidates. These
        # are normal local algorithms, locking, and atomic storage operations.
        patterns = [
            "import numpy as np\nfrom sentence_transformers import SentenceTransformer\n"
            "class NewProvider:\n    def norm(self, a):\n        return float(np.linalg.norm(a))\n",
            "import threading\nimport os\nclass NewProvider:\n"
            "    def save(self, tmp_path, database_path):\n"
            "        with threading.RLock():\n            os.replace(tmp_path, database_path)\n",
            "import os\nclass NewProvider:\n"
            "    def format(self, rows):\n        return os.linesep.join(rows)\n",
        ]
        for code in patterns:
            with self.subTest(code=code):
                self.generator.proposal["provider_code"]["code"] = code
                self.assertEqual(self.project()["provider_code"]["code"], code)

    def test_native_local_policy_allows_normal_stdlib_and_local_reflection(self):
        self.install(code=True, generated_code_policy="native_local", compact_provider=False)
        code = (
            "from concurrent.futures import ThreadPoolExecutor\n"
            "from decimal import Decimal\nfrom os import replace as atomic_replace\n"
            "from pathlib import Path\nimport sqlite3\nimport threading\n"
            "class NewProvider:\n"
            "    def run(self, response, field, tmp_path, database_path):\n"
            "        value = getattr(response, field, None)\n"
            "        setattr(self, 'last_value', value)\n"
            "        atomic_replace(tmp_path, database_path)\n"
            "        return Decimal('1.25')\n")
        self.generator.proposal["provider_code"]["code"] = code
        self.assertEqual(self.project()["provider_code"]["code"], code)
        prompt = self.generator._build_generation_prompt()
        self.assertIn("available local algorithm libraries", prompt)
        self.assertIn("native validator", prompt)
        self.assertNotIn("Use os.path, os.makedirs and JSON files", prompt)
        self.assertNotIn("220 Python lines", prompt)

    def test_native_local_policy_blocks_concrete_external_actions_and_secret_access(self):
        self.install(code=True, generated_code_policy="native_local")
        dangerous = [
            "import requests", "import subprocess", "import urllib.request", "import socket",
            "from concurrent.futures import ProcessPoolExecutor",
            "import asyncio\nasyncio.create_subprocess_exec('command')",
            "from os import system as shell", "import os\nos.system('command')",
            "import os as alias\nalias.environ['OPENAI_API_KEY']",
            "from os import getenv as read_secret", "import os\ngetattr(os, 'getenv')('KEY')",
            "import os\nos.posix_spawn('command', [], {})", "import ctypes", "import dotenv",
            "eval('anything')", "exec('anything')", "from builtins import eval as execute",
            "__import__('json')", "import importlib", "OpenAIServerModel()",
            "from pathlib import Path\nPath('../outside').write_text('anything')",
        ]
        for operation in dangerous:
            with self.subTest(operation=operation):
                self.generator.proposal["provider_code"]["code"] = operation + "\nclass NewProvider: pass\n"
                with self.assertRaises(ValueError):
                    self.project()

    def test_native_local_allows_only_literal_offline_flag_defaults_with_import_aliases(self):
        self.install(code=True, generated_code_policy="native_local")
        for prefix, name in (("import os", "os.environ"),
                             ("import os as local_os", "local_os.environ"),
                             ("from os import environ as local_env", "local_env")):
            with self.subTest(prefix=prefix):
                code = (prefix + "\n" + name + ".setdefault('HF_HUB_OFFLINE', '1')\n"
                        + name + ".setdefault('TRANSFORMERS_OFFLINE', '1')\nclass NewProvider: pass\n")
                self.generator.proposal["provider_code"]["code"] = code
                self.assertEqual(self.project()["provider_code"]["code"], code)

    def test_native_local_offline_flag_exception_does_not_allow_other_environment_access(self):
        self.install(code=True, generated_code_policy="native_local")
        operations = [
            "os.environ.setdefault('OPENAI_API_KEY', '1')",
            "os.environ.setdefault('HF_HUB_OFFLINE', '0')",
            "os.environ.setdefault('HF_HUB_OFFLINE', 1)",
            "os.environ.setdefault(key, '1')", "os.environ.setdefault('HF_HUB_OFFLINE', value)",
            "os.environ['HF_HUB_OFFLINE']", "os.environ['HF_HUB_OFFLINE'] = '1'",
            "os.environ.get('HF_HUB_OFFLINE')", "fn = os.environ.setdefault",
            "getattr(os, 'environ').setdefault('HF_HUB_OFFLINE', '1')",
            "env = os.environ\nenv.setdefault('HF_HUB_OFFLINE', '1')",
        ]
        for operation in operations:
            for prefix, expression in (("import os", operation),
                                       ("import os as alt", operation.replace("os.", "alt.")),
                                       ("from os import environ as alt", operation.replace("os.environ", "alt"))):
                with self.subTest(operation=operation, prefix=prefix):
                    self.generator.proposal["provider_code"]["code"] = prefix + "\n" + expression + "\nclass NewProvider: pass\n"
                    with self.assertRaises(ValueError):
                        self.project()

    def test_native_local_preserves_existing_factor_and_storage_isolation_checks(self):
        self.install(code=True, generated_code_policy="native_local")
        self.generator.proposal["config_updates"]["storage_dir"] = "'../outside'"
        with self.assertRaisesRegex(ValueError, "escapes workspace/storage"):
            self.project()
        self.generator.proposal = self.proposal()
        projected = self.project()
        self.assertEqual(projected["orientation_items"], list(self.snapshot.items))
        event = json.loads((self.run / "evidence/factor_projection.jsonl").read_text())
        self.assertEqual(event["generated_code_policy"], "native_local")

    def test_unknown_codegen_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown generated provider policy"):
            self.install(generated_code_policy="typo")

    def test_frozen_orientation_projects_away_invalid_disabled_proposal(self):
        self.install()
        self.generator.proposal["orientation_items"] = [""]
        self.assertEqual(self.project()["orientation_items"], list(self.snapshot.items))

    def test_dangerous_imports_reflection_and_shell_actions_rejected(self):
        self.install(code=True)
        dangerous = ["import requests", "import subprocess", "import urllib.request", "import socket",
                     "from os import system", "import os\nos.system('anything')",
                     "import os as alias\nalias.popen('anything')",
                     "eval('anything')", "__import__('json')", "import importlib",
                     "import logging.handlers", "from logging import handlers",
                     "super().__class__", "super().__dict__", "super(X, self).__init__()",
                     "super(type=X, obj=self).__init__()", "BaseMemoryProvider.__init__(self)",
                     "getattr(response, field_name)", "getattr(response, '__dict__')",
                     "getattr(response, 'config')", "getattr(response, 'content', None, 4)",
                     "getattr(response, name='content')",
                     "class X: pass\nX.__class__", "from builtins import open"]
        for operation in dangerous:
            with self.subTest(operation=operation):
                self.generator.proposal["provider_code"]["code"] = operation + "\nclass NewProvider:\n    pass\n"
                with self.assertRaises(ValueError):
                    self.project()

    def test_normal_super_constructor_is_allowed_without_enabling_reflection(self):
        # Both failed real pilot candidates use this normal inherited constructor.
        self.install(code=True)
        for enum_name in ("DIALECTIC_CALIBRATION_MEMORY", "DIALECTIC_PITFALL_FORGE"):
            with self.subTest(enum_name=enum_name):
                self.generator.proposal["provider_code"]["code"] = (
                    "from EvolveLab.base_memory import BaseMemoryProvider\n"
                    "from EvolveLab.memory_types import MemoryType\n"
                    "class NewProvider(BaseMemoryProvider):\n"
                    "    def __init__(self, config=None):\n"
                    f"        super().__init__(memory_type=MemoryType.{enum_name}, config=config or {{}})\n"
                )
                projected = self.project()
                self.assertEqual(projected["provider_code"], self.generator.proposal["provider_code"])

    def test_standard_model_response_fields_allow_constant_getattr_only(self):
        self.install(code=True)
        for field in ("content", "reasoning_content", "tool_calls"):
            for default in ("", ", str(response)"):
                with self.subTest(field=field, default=default):
                    self.generator.proposal["provider_code"]["code"] = (
                        "class NewProvider:\n"
                        "    def response_field(self, response):\n"
                        f"        return getattr(response, {field!r}{default})\n"
                    )
                    projected = self.project()
                    self.assertEqual(projected["provider_code"], self.generator.proposal["provider_code"])

    def test_storage_escape_and_existing_module_or_enum_rejected(self):
        self.install(code=True)
        for path in ("../outside", "/outside", "./storage/../../outside", "C:\\private\\state.json"):
            with self.subTest(path=path):
                self.generator.proposal["config_updates"]["storage_dir"] = repr(path)
                with self.assertRaisesRegex(ValueError, "escapes workspace/storage"):
                    self.project()
        self.generator.proposal = self.proposal()
        existing = self.workspace / "EvolveLab/providers/new_provider.py"
        existing.write_text("existing bytes", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "module already exists"):
            self.project()
        self.assertEqual(existing.read_text(), "existing bytes")
        existing.unlink()
        self.generator.proposal["memory_type_info"]["enum_name"] = "LIGHTWEIGHT_MEMORY"
        with self.assertRaisesRegex(ValueError, "enum already exists"):
            self.project()

    def test_uniform_prompt_and_idempotent_restore(self):
        original = self.generator_class._generate_single_system
        restore = self.install()
        self.assertIs(install_native_policy(self.run, self.job), restore)
        prompt = self.generator._build_generation_prompt()
        for required in ("config.get('model')", "retain, revise, or clear", "change is not required", "./storage/<your enum_value>/"):
            self.assertIn(required, prompt)
        for label in ("engineering", "dialectical", "critical", "heldout"):
            self.assertNotIn(label, prompt)
        with self.assertRaisesRegex(ValueError, "different ablation policy"):
            install_native_policy(self.run, dict(self.job, code_evolve=True))
        restore()
        restore()
        self.assertIs(self.generator_class._generate_single_system, original)


if __name__ == "__main__":
    unittest.main()
