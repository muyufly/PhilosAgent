"""Offline tests of the patched upstream evolution boundaries; no model calls."""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from philoagent.orientation import bootstrap, fork, load


UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "MemEvolve" / "Flash-Searcher-main"


def load_upstream(relative, name):
    """Load the real patched file, replacing only unused heavyweight dependencies."""
    modules = {}
    for package, directory in {
        "MemEvolve": "MemEvolve", "MemEvolve.core": "MemEvolve/core",
        "MemEvolve.phases": "MemEvolve/phases", "MemEvolve.utils": "MemEvolve/utils",
        "EvolveLab": "EvolveLab",
    }.items():
        module = types.ModuleType(package)
        module.__path__ = [str(UPSTREAM / directory)]
        modules[package] = module
    for module_name, symbol in {
        "base_agent": "AnalysisAgent",
        "MemEvolve.core.memory_evolver": "MemoryEvolver",
        "MemEvolve.utils.trajectory_tools": "TrajectoryFeedbackAggregator",
    }.items():
        module = types.ModuleType(module_name)
        setattr(module, symbol, object)
        modules[module_name] = module
    for module_name, symbols in {
        "dotenv": {"load_dotenv": lambda **kwargs: None},
        "openai": {"OpenAI": object},
        "yaml": {"safe_load": lambda value: {}},
    }.items():
        module = types.ModuleType(module_name)
        module.__dict__.update(symbols)
        modules[module_name] = module
    spec = importlib.util.spec_from_file_location(name, UPSTREAM / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


AUTO = load_upstream("MemEvolve/core/auto_evolver.py", "MemEvolve.core.auto_evolver")
ANALYZER = load_upstream("MemEvolve/phases/phase_analyzer.py", "MemEvolve.phases.phase_analyzer")
GENERATOR = load_upstream("MemEvolve/phases/phase_generator.py", "MemEvolve.phases.phase_generator")
CREATOR = load_upstream("MemEvolve/phases/memory_creator.py", "MemEvolve.phases.memory_creator")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.lineage = self.directory / "lineage"
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("PHILO_LINEAGE_ROOT", None)
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def activate(self, items=("Revise beliefs using evidence.",)):
        bootstrap(self.lineage, "parent", list(items))
        os.environ["PHILO_LINEAGE_ROOT"] = str(self.lineage)
        return load(self.lineage, "parent")

    def generated_config(self, child="child", **proposal):
        config = {
            "provider_code": {"class_name": "ChildProvider", "module_name": f"{child}_provider",
                              "code": "class ChildProvider:\n    pass\n"},
            "config_updates": {"top_k": "3"},
            "memory_type_info": {"enum_name": child.upper(), "enum_value": child},
        }
        if os.getenv("PHILO_LINEAGE_ROOT"):
            config["_orientation_parent"] = {
                "lineage_root": str(self.lineage.resolve()),
                "snapshot": load(self.lineage, "parent").to_dict(),
            }
        config.update(proposal)
        return config

    def fixture_project(self):
        project = self.directory / "project"
        package = project / "EvolveLab"
        package.mkdir(parents=True)
        (package / "memory_types.py").write_text(
            'from enum import Enum\nclass MemoryType(Enum):\n    BASE = "base"\n'
            '    # add new memory type upside this line(Enum)\n'
            'PROVIDER_MAPPING = {\n# add new memory type upside this line(PROVIDER_MAPPING)\n}\n',
            encoding="utf-8",
        )
        (package / "config.py").write_text(
            "providers = {\n# add new memory type upside this line\n}\n", encoding="utf-8"
        )
        return project

    def test_creator_inherits_clears_and_replays_without_reseeding(self):
        parent = self.activate()
        project = self.fixture_project()
        inherited = self.generated_config()
        self.assertTrue(CREATOR.MemorySystemCreator.create_memory_system(inherited, project)["success"])
        self.assertEqual(load(self.lineage, "child").items, parent.items)
        self.assertTrue(CREATOR.MemorySystemCreator.create_memory_system(inherited, project)["success"])
        cleared = self.generated_config("cleared", orientation_items=[])
        self.assertTrue(CREATOR.MemorySystemCreator.create_memory_system(cleared, project)["success"])
        self.assertEqual(load(self.lineage, "cleared").items, ())
        config_source = (project / "EvolveLab/config.py").read_text(encoding="utf-8")
        self.assertNotIn("orientation_items", config_source)
        self.assertEqual(load(self.lineage, "parent").items, parent.items)

    def test_creator_rejects_invalid_or_cross_run_state_before_writing_code(self):
        self.activate()
        project = self.fixture_project()
        bad = self.generated_config(orientation_items=None)
        self.assertFalse(CREATOR.MemorySystemCreator.create_memory_system(bad, project)["success"])
        self.assertFalse((project / "EvolveLab/providers/child_provider.py").exists())
        bad = self.generated_config()
        bad["_orientation_parent"]["snapshot"]["items"] = ["stale"]
        self.assertFalse(CREATOR.MemorySystemCreator.create_memory_system(bad, project)["success"])
        self.assertFalse((project / "EvolveLab/providers/child_provider.py").exists())
        other_root = self.directory / "other_lineage"
        bootstrap(other_root, "parent", ["Revise beliefs using evidence."])
        config = self.generated_config()
        os.environ["PHILO_LINEAGE_ROOT"] = str(other_root)
        self.assertFalse(CREATOR.MemorySystemCreator.create_memory_system(config, project)["success"])

    def test_legacy_creator_needs_no_orientation(self):
        project = self.fixture_project()
        self.assertTrue(CREATOR.MemorySystemCreator.create_memory_system(self.generated_config(), project)["success"])
        self.assertFalse(self.lineage.exists())

    def generator(self):
        generator = GENERATOR.PhaseGenerator(None, "offline-test", self.directory, "parent")
        generator._load_prompt_template = lambda: "Template {default_provider}\n{analysis_section}"
        generator._get_existing_system_names = lambda: []
        return generator

    def model_response(self, orientation_line=""):
        return (
            "## Provider Information\n**Class Name**: ChildProvider\n**Module Name**: child_provider\n"
            "## Provider Code\n```python\nclass ChildProvider:\n    pass\n```\n"
            "## Configuration\n**top_k**: 3\n" + orientation_line +
            "\n## Memory Type\n**Enum Name**: CHILD\n**Enum Value**: child\n"
        )

    def test_parser_distinguishes_missing_empty_and_revised_proposals(self):
        generator = self.generator()
        self.assertNotIn("orientation_items", generator._parse_system_config(self.model_response()))
        cleared = generator._parse_system_config(self.model_response("**orientation_items**: []"))
        self.assertEqual(cleared["orientation_items"], [])
        revised = generator._parse_system_config(self.model_response('**orientation_items**: ["Evidence (including failures)."]'))
        self.assertEqual(revised["orientation_items"], ["Evidence (including failures)."])
        self.assertNotIn("orientation_items", revised["config_updates"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(generator._parse_system_config(self.model_response("**orientation_items**: null")))

    def test_analyzer_and_generator_read_current_empty_snapshot(self):
        self.activate(["Original seed."])
        fork(self.lineage, "parent", "empty_parent", [])
        analyzer = ANALYZER.PhaseAnalyzer.__new__(ANALYZER.PhaseAnalyzer)
        analyzer._load_prompt_template = lambda: "Analysis {default_provider} {overview}"
        analysis_prompt = analyzer._build_analysis_prompt({"task_summaries": [], "total_tasks": 0}, "empty_parent", ".")
        generator = self.generator()
        generator.default_provider = "empty_parent"
        generation_prompt = generator._build_generation_prompt("analysis", "", 0.5)
        for prompt in (analysis_prompt, generation_prompt):
            self.assertIn('"items": []', prompt)
            self.assertIn('"provider_id": "empty_parent"', prompt)
            self.assertNotIn("Original seed.", prompt)
        self.assertIn("config['philo_orientation_items']", generation_prompt)
        self.assertIn("trajectory_data.metadata['philo_orientation_items']", generation_prompt)

    def test_generation_records_exact_snapshot_and_rejects_stale_analysis(self):
        snapshot = self.activate()
        generator = self.generator()
        generator._get_memory_types_reference = lambda: ""
        completion = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.model_response()))])
        generator.openai_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=Mock(return_value=completion))))
        with self.assertRaisesRegex(ValueError, "Analysis orientation snapshot"):
            generator.run_generation({"agent_result": "stale"})
        self.assertFalse(generator.openai_client.chat.completions.create.called)
        result = generator.run_generation({"agent_result": "analysis", "orientation_snapshot": snapshot.to_dict()})
        self.assertTrue(result["success"])
        self.assertEqual(result["config"]["_orientation_parent"]["snapshot"], snapshot.to_dict())
        self.assertEqual(result["config"]["_generation_metadata"]["parent_provider"], "parent")

    def make_auto(self, count=2):
        return AUTO.AutoEvolver(
            analysis_model_id="offline", gen_model_id="offline", work_root=str(self.directory / "run"),
            dataset_name="fixture", datasets_config={"fixture": str(self.directory / "data.jsonl")},
            run_provider=lambda *args: args[-1], default_provider="initial", num_systems=count,
            clear_storage_per_round=False,
        )

    def fake_evolver(self, initial_parent="initial", existing=0):
        directory = self.directory / "generated"
        directory.mkdir(exist_ok=True)
        state = {"phases": {"analyze": {"default_provider": initial_parent, "completed": True}}}
        fake = types.SimpleNamespace(state=state, analyzed=[], generated=[], configs=[])
        fake._save_state = Mock()
        for i in range(existing):
            path = directory / f"generated_system_{i}.json"
            path.write_text(json.dumps({"memory_type_info": {"enum_value": f"child{i}"}}), encoding="utf-8")
            fake.configs.append(path)
        fake._get_all_generated_system_paths = lambda: fake.configs.copy()

        def analyze(**kwargs):
            fake.analyzed.append(kwargs["default_provider"])
            fake.state["phases"]["analyze"]["default_provider"] = kwargs["default_provider"]
            return {"success": True}

        def generate(**kwargs):
            fake.generated.append(fake.state["phases"]["analyze"]["default_provider"])
            name = f"child{len(fake.configs)}"
            config = {"memory_type_info": {"enum_value": name}}
            path = directory / f"generated_system_{len(fake.configs)}.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            fake.configs.append(path)
            return {"success": True, "config": config, "config_path": str(path)}

        fake.analyze = analyze
        fake.generate = generate
        fake.create = lambda config_path: {"success": True, "created": [json.loads(Path(config_path).read_text())["memory_type_info"]["enum_value"]]}
        fake.validate = lambda **kwargs: {"success": True, "validated": kwargs["created_systems"]}
        return fake

    def test_current_parent_reaches_initial_and_repeated_analysis(self):
        auto = self.make_auto()
        fake = self.fake_evolver()
        with patch.object(AUTO, "MemoryEvolver", return_value=fake):
            auto._run_memory_evolver(self.directory, current_provider="winner")
        self.assertEqual(fake.analyzed, ["winner", "winner"])
        self.assertEqual(fake.generated, ["winner", "winner"])

    def test_resume_skips_only_analysis_for_the_correct_parent(self):
        for saved_parent, expected_calls in (("winner", []), ("initial", ["winner"])):
            with self.subTest(saved_parent=saved_parent):
                auto = self.make_auto(1)
                fake = self.fake_evolver(initial_parent=saved_parent)
                checkpoint = {"parent_provider": "winner", "evolution_completed": {"analysis": True}}
                with patch.object(AUTO, "MemoryEvolver", return_value=fake):
                    auto._run_memory_evolver(self.directory, checkpoint, "winner")
                self.assertEqual(fake.analyzed, expected_calls)
                self.assertEqual(fake.generated, ["winner"])

    def test_partial_resume_generates_all_remaining_candidates(self):
        auto = self.make_auto(3)
        fake = self.fake_evolver(initial_parent="winner", existing=1)
        checkpoint = {"parent_provider": "winner", "validated_systems": ["child0"]}
        with patch.object(AUTO, "MemoryEvolver", return_value=fake):
            result = auto._run_memory_evolver(self.directory, checkpoint, "winner")
        self.assertEqual(len(fake.generated), 2)
        self.assertEqual(result["validated_systems"], ["child0", "child1", "child2"])

    def test_checkpoint_parent_conflict_fails_before_model_work(self):
        auto = self.make_auto(1)
        with patch.object(AUTO, "MemoryEvolver") as constructor:
            with self.assertRaisesRegex(ValueError, "Checkpoint parent"):
                auto._run_memory_evolver(self.directory, {"parent_provider": "other"}, "winner")
            constructor.assert_not_called()

    def test_winner_code_and_orientation_become_next_round_parent(self):
        bootstrap(self.lineage, "initial", ["Original orientation."])
        fork(self.lineage, "initial", "child0", [])
        fork(self.lineage, "child0", "child1", ["Revised orientation."])
        os.environ["PHILO_LINEAGE_ROOT"] = str(self.lineage)
        auto = self.make_auto(1)
        auto.task_batch_x = 1
        auto.extra_sample_y = 0
        auto.top_t = 1
        auto._select_tasks = lambda start, count: list(range(start, start + count))
        parents = []

        def evolve(round_dir, checkpoint, current_provider):
            parents.append(current_provider)
            self.assertEqual(checkpoint["parent_orientation_sha256"], load(self.lineage, current_provider).sha256)
            return {"validated_systems": [f"child{len(parents) - 1}"]}

        auto._run_memory_evolver = evolve
        auto._evaluate_providers_parallel = lambda **kwargs: {p: {} for p in kwargs["providers"]}
        auto._select_top = lambda results, k: list(results)[-k:]
        result = auto.run(num_rounds=2)
        self.assertEqual(parents, ["initial", "child0"])
        self.assertEqual(auto.state["best_provider"], "child1")
        self.assertEqual(auto.state["best_orientation_sha256"], load(self.lineage, "child1").sha256)
        self.assertEqual(result["history"][0]["winner_orientation_snapshot"]["items"], [])


if __name__ == "__main__":
    unittest.main()
