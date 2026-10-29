"""Exercise native actor/runner exception boundaries without models or network."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


# Native imports must not discover or read the user's .env file.
with patch.dict(os.environ, {"PHILO_LINEAGE_ROOT": "offline-import-only"}):
    from FlashOAgents.agents import ToolCallingAgent
    from EvolveLab.memory_types import MemoryResponse, MemoryStatus, MemoryType
    from base_agent import BaseAgent
    import run_flash_searcher_webwalkerqa as runner

from philoagent.orientation import OrientationIntegrityError, bootstrap, load
from philoagent.runtime import OrientedProvider


class ExperimentalFailClosedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ["PHILO_LINEAGE_ROOT"] = str(self.root / "lineage")
        os.environ.pop("PHILO_EVIDENCE_DIR", None)

    def actor(self):
        path = bootstrap(self.root / "lineage", "base", ["A provisional method."])
        snapshot = load(self.root / "lineage", "base")
        code = self.root / "provider.py"
        code.write_text("# offline provider\n", encoding="utf-8")
        native = Mock()
        native.provide_memory.return_value = MemoryResponse([], MemoryType.LIGHTWEIGHT_MEMORY, 0)
        wrapper = OrientedProvider(native, snapshot, code, hashlib.sha256(code.read_bytes()).hexdigest())
        actor = ToolCallingAgent.__new__(ToolCallingAgent)
        actor.memory_provider = wrapper
        actor.task = "Offline task"
        actor._format_current_context = lambda: ""
        return actor, native, path, code

    def test_actual_actor_propagates_orientation_integrity_failure(self):
        actor, native, path, _ = self.actor()
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["items"] = ["Tampered current method."]
        path.write_text(json.dumps(saved), encoding="utf-8")
        with self.assertRaises(OrientationIntegrityError):
            actor._get_memory_guidance(MemoryStatus.BEGIN)
        native.provide_memory.assert_not_called()

    def test_actual_actor_propagates_loaded_code_integrity_failure(self):
        actor, native, _, code = self.actor()
        code.write_text("# changed after loading\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            actor._get_memory_guidance(MemoryStatus.BEGIN)
        native.provide_memory.assert_not_called()

    def test_unconfigured_actor_preserves_upstream_error_tolerance(self):
        actor, native, _, code = self.actor()
        code.write_text("# changed after loading\n", encoding="utf-8")
        os.environ.pop("PHILO_LINEAGE_ROOT", None)
        self.assertIsNone(actor._get_memory_guidance(MemoryStatus.BEGIN))
        native.provide_memory.assert_not_called()

    def process(self, *, ingestion=None, actor_error=None):
        provider = Mock()
        provider.take_in_memory.return_value = (True, "stored")
        if isinstance(ingestion, Exception):
            provider.take_in_memory.side_effect = ingestion
        elif ingestion is not None:
            provider.take_in_memory.return_value = ingestion
        search = BaseAgent.__new__(BaseAgent)
        search.agent_fn = Mock()
        search.agent_fn.write_memory_to_messages.return_value = []
        if actor_error:
            search.agent_fn.run.side_effect = actor_error
        else:
            search.agent_fn.run.return_value = "Offline answer"
        search.capture_trajectory = lambda: {"agent_trajectory": []}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "OpenAIServerModel", return_value=Mock()))
            stack.enter_context(patch.object(runner, "SearchAgent", return_value=search))
            stack.enter_context(patch.object(runner, "load_memory_provider", return_value=provider))
            stack.enter_context(patch.object(runner.TokenCounter, "from_model", return_value=Mock()))
            stack.enter_context(patch.object(runner, "enrich_result_with_metrics", side_effect=lambda result, *args: result))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return runner.process_item(
                {"question": "Offline question", "answer": "Offline answer", "root_url": "https://example.invalid"},
                {}, 10, "default", 2, memory_type_str="base", judge_model=None,
            )

    def test_runner_rejects_actor_error_returned_after_native_retry_loop(self):
        with self.assertRaisesRegex(RuntimeError, "Actor failed"):
            self.process(actor_error=OrientationIntegrityError("Orientation changed during task"))

    def test_runner_propagates_ingestion_exception(self):
        with self.assertRaises(OrientationIntegrityError):
            self.process(ingestion=OrientationIntegrityError("Orientation changed during task"))

    def test_runner_rejects_explicit_ingestion_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Memory ingestion failed"):
            self.process(ingestion=(False, "could not store experience"))

    def test_unconfigured_runner_preserves_upstream_ingestion_tolerance(self):
        os.environ.pop("PHILO_LINEAGE_ROOT", None)
        self.assertEqual(self.process(ingestion=RuntimeError("legacy storage unavailable"))["status"], "success")

    def run_batch(self, *, provider_error=None):
        args = types.SimpleNamespace(
            infile="offline.jsonl", outfile=str(self.root / "results.jsonl"),
            difficulty=None, lang=None, domain=None, question_type=None,
            task_indices=None, sample_num=None, memory_provider="base",
            direct_output_dir=str(self.root / "results"), concurrency=1,
            summary_interval=10, prompts_type="default", max_steps=2,
            enable_memory_evolution=True, judge_model=None,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "OpenAIServerModel", return_value=Mock()))
            stack.enter_context(patch.object(runner, "read_jsonl", return_value=[{"question": "Offline task"}]))
            provider_loader = stack.enter_context(patch.object(runner, "load_memory_provider"))
            provider_loader.return_value.get_memory_type.return_value.value = "base"
            if provider_error:
                provider_loader.side_effect = provider_error
            stack.enter_context(patch.object(runner, "process_item", side_effect=OrientationIntegrityError("Invalid worker state")))
            output = stack.enter_context(patch.object(runner, "write_jsonl"))
            self.batch_output = output
            stack.enter_context(patch.object(runner, "generate_unified_report"))
            runner.main(args)
            return output

    def test_batch_runner_propagates_worker_failure_without_exporting_a_completed_batch(self):
        with self.assertRaises(OrientationIntegrityError):
            self.run_batch()
        self.batch_output.assert_not_called()

    def test_batch_runner_propagates_preflight_provider_failure(self):
        with self.assertRaises(OrientationIntegrityError):
            self.run_batch(provider_error=OrientationIntegrityError("Invalid preflight state"))

    def test_unconfigured_batch_runner_preserves_upstream_worker_tolerance(self):
        os.environ.pop("PHILO_LINEAGE_ROOT", None)
        self.run_batch().assert_called_once_with(str(self.root / "results.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
