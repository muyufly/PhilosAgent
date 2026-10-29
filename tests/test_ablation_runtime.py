"""Offline execution-boundary checks. No .env reads and no network requests."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from philoagent.ablation_runtime import (
    AuditSink, AuditedClient, AuditedRequestError, EmptyModelContentError,
    NativeTaskRunner, audit_scope, collect_completed_development,
    digest, items_digest, reset_storage, write_json,
)
from philoagent.orientation import bootstrap


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def result(self, content="answer", tools=None):
        message = SimpleNamespace(content=content, tool_calls=tools)
        return SimpleNamespace(id="fake-response", choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5))

    def sink(self, provider="parent", items=None):
        if items is None:
            items = ['依据“证据”修正方法。']
        bootstrap(self.root / "lineage", provider, items)
        return AuditSink(self.root / "api.jsonl", self.root / "lineage", model="deepseek-flash")


class AuditedClientTests(RuntimeFixture):
    def test_sdk_request_is_capped_and_chinese_principle_matches_actual_strings(self):
        sink = self.sink()
        raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=self.result()))))
        with audit_scope(provider="parent", role="actor", stage="batch_comparison"):
            client = AuditedClient(raw, sink)
        client.create(model="wrong-default", messages=[{"role": "user", "content": '依据“证据”修正方法。'}],
                      max_tokens=30000, max_completion_tokens=32768)
        kwargs = raw.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-flash")
        self.assertEqual(kwargs["max_tokens"], 8192)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})
        event = sink.events[0]
        self.assertTrue(event["orientation_present_in_request"])
        self.assertEqual(event["orientation_items_sha256"], items_digest(['依据“证据”修正方法。']))
        self.assertEqual(event["finish_reason"], ["stop"])
        self.assertEqual(sink.summary()["roles"]["actor"]["total_tokens"], 5)
        self.assertNotIn('依据“证据”修正方法。', sink.path.read_text(encoding="utf-8"))

    def test_with_options_preserves_audit_and_native_lower_limit(self):
        sink = self.sink()
        raw = Mock()
        raw.with_options.return_value = raw
        raw.chat.completions.create.return_value = self.result()
        with audit_scope(provider="parent", role="generation"):
            client = AuditedClient(raw, sink).with_options(timeout=900, max_retries=8)
        client.create(messages=[], max_tokens=14000)
        raw.with_options.assert_called_once_with(timeout=120, max_retries=0)
        self.assertEqual(raw.chat.completions.create.call_args.kwargs["max_tokens"], 14000)
        self.assertEqual(sink.events[0]["role"], "generation")

    def test_empty_body_is_failure_but_tool_response_is_usable(self):
        sink = self.sink(items=[])
        raw = Mock()
        raw.chat.completions.create.side_effect = [self.result(content=" "), self.result(content=None, tools=[{"name": "search"}])]
        with audit_scope(provider="parent", role="actor"):
            client = AuditedClient(raw, sink)
            with self.assertRaises(EmptyModelContentError):
                client.create(messages=[])
            client.create(messages=[])
        self.assertEqual([event["status"] for event in sink.events], ["error", "success"])
        self.assertFalse(sink.events[1]["orientation_present_in_request"])

    def test_exception_text_is_not_saved_or_rethrown(self):
        sink = self.sink()
        raw = Mock()
        raw.chat.completions.create.side_effect = ValueError("Bearer secret-key-in-error")
        with audit_scope(provider="parent", role="judge"):
            with self.assertRaises(AuditedRequestError) as error:
                AuditedClient(raw, sink).create(messages=[])
        self.assertNotIn("secret-key", str(error.exception))
        self.assertNotIn("secret-key", sink.path.read_text(encoding="utf-8"))
        self.assertEqual(sink.summary()["failed_request_count"], 1)

    def test_parallel_provider_contexts_do_not_cross(self):
        sink = self.sink(provider="parent", items=["parent method"])
        bootstrap(self.root / "lineage", "child", ["child method"])
        barrier = threading.Barrier(2)
        raw = Mock()
        raw.chat.completions.create.return_value = self.result()
        client = AuditedClient(raw, sink)
        def call(provider):
            with audit_scope(provider=provider, role="actor", stage=provider):
                barrier.wait(timeout=5)
                client.create(messages=[{"content": provider + " method"}])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(call, ["parent", "child"]))
        self.assertEqual({event["provider"] for event in sink.events}, {"parent", "child"})
        self.assertTrue(all(event["orientation_present_in_request"] and event["stage"] == event["provider"] for event in sink.events))


class FairStorageTests(RuntimeFixture):
    def test_reset_actual_effective_directory_and_reject_outside_paths(self):
        storage = self.workspace / "storage" / "parent"
        storage.mkdir(parents=True)
        (storage / "memory.json").write_text('{"old": true}', encoding="utf-8")
        result = reset_storage(self.workspace, "parent", {"storage_dir": "storage/parent"})
        self.assertTrue(result["reset"])
        self.assertEqual(result["previous_file_count"], 1)
        self.assertEqual(list(storage.iterdir()), [])
        for config in ({}, {"storage_dir": "storage"}, {"storage_dir": str(self.root)},
                       {"storage_dir": "storage/child"},
                       {"storage_dir": "storage/parent", "db_path": "storage/other/memory.json"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                reset_storage(self.workspace, "parent", config)

    def test_completed_warmup_uses_discovery_and_winner_new_task(self):
        evolution = self.root / "evolution"
        for round_number in range(2):
            root = evolution / f"round_{round_number:02d}"
            batch = round_number * 2
            write_json(root / "base_logs" / f"{batch + 1}.json", {
                "global_task_index": batch, "origin": "discovery"})
            write_json(root / "eval_finalists" / "winner" / f"{batch + 1}.json", {
                "global_task_index": batch, "origin": "repeated-winner-task"})
            write_json(root / "eval_finalists" / "winner" / f"{batch + 2}.json", {
                "global_task_index": batch + 1, "origin": "winner-new-task"})
            write_json(root / "eval_finalists" / "loser" / f"{batch + 2}.json", {
                "global_task_index": batch + 1, "origin": "loser"})
            write_json(root / "round_summary.json", {"round": round_number, "batch_tasks": [batch],
                       "finalist_tasks": [batch, batch + 1], "winner": "winner"})
        records = collect_completed_development(evolution)
        self.assertEqual([item["global_task_index"] for item in records], [0, 1, 2, 3])
        self.assertEqual([item["data"]["origin"] for item in records],
                         ["discovery", "winner-new-task", "discovery", "winner-new-task"])
        self.assertEqual(len(collect_completed_development(evolution, before_round=1)), 2)
        self.assertTrue(all(item["source_sha256"] == digest(Path(item["source"]).read_bytes()) for item in records))


class NativeCallbackTests(RuntimeFixture):
    def make_callback(self):
        import enum
        class MemoryType(enum.Enum):
            PARENT = "parent"
            CHILD = "child"
        modules = {}
        package = ModuleType("EvolveLab")
        package.__path__ = []
        modules["EvolveLab"] = package
        types = ModuleType("EvolveLab.memory_types")
        types.MemoryType = MemoryType
        types.PROVIDER_MAPPING = {MemoryType.PARENT: ("Provider", "parent_provider"), MemoryType.CHILD: ("Provider", "child_provider")}
        types.TrajectoryData = lambda **kwargs: SimpleNamespace(**kwargs)
        modules[types.__name__] = types
        config = ModuleType("EvolveLab.config")
        config.get_memory_config = lambda kind: {"storage_dir": "storage/" + kind.value}
        modules[config.__name__] = config
        for provider in ("parent", "child"):
            module = ModuleType("EvolveLab.providers." + provider + "_provider")
            source = self.workspace / "EvolveLab" / "providers" / f"{provider}_provider.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("class Provider: pass\n", encoding="utf-8")
            module.__file__ = str(source)
            module.Provider = object
            modules[module.__name__] = module
            bootstrap(self.root / "lineage", provider, ["method"])
        self.modules = patch.dict("sys.modules", modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.ingested = []
        self.processed = []
        def loader(provider, model):
            def ingest(data):
                self.ingested.append((provider, data))
                return True, "ok"
            return SimpleNamespace(take_in_memory=ingest)
        def process(item, model_config, **kwargs):
            self.processed.append((item, kwargs))
            return {"status": "success", "judgement": "correct", "question": item["question"],
                    "agent_result": "answer", "agent_messages": [{"role": "assistant", "content": "trace"}]}
        runner = SimpleNamespace(load_memory_provider=loader, process_item=process)
        native = SimpleNamespace(OpenAIServerModel=lambda **kwargs: object())
        return NativeTaskRunner(self.root, self.workspace, {"max_steps": 6}, {"model_id": "mock"}, runner=runner, native=native)

    def data(self):
        path = self.root / "dev.jsonl"
        path.write_text("".join(json.dumps({"question": f"q{index}", "answer": "gold", "root_url": "https://example.org", "task_id": f"task-{index}"}) + "\n" for index in range(4)), encoding="utf-8")
        return path

    def test_native_callback_order_learning_and_matched_warmup(self):
        callback = self.make_callback()
        path = self.data()
        root = self.root / "evolution" / "round_00"
        callback("webwalkerqa", "parent", [0], path, root / "base_logs")
        callback("webwalkerqa", "parent", [0, 1], path, root / "eval_finalists" / "parent")
        callback("webwalkerqa", "child", [0, 1], path, root / "eval_finalists" / "child")
        self.assertEqual([call[1]["item_index"] for call in self.processed], [0, 0, 1, 0, 1])
        self.assertTrue(all(call[1]["enable_memory_evolution"] for call in self.processed))
        self.assertEqual([provider for provider, record in self.ingested], ["parent", "child"])
        self.assertTrue(all(record.query == "q0" and "gold" not in str(record.trajectory) for _, record in self.ingested))
        events = callback.stage_events
        self.assertEqual(events[1]["warmup"]["warmup_sha256"], events[2]["warmup"]["warmup_sha256"])
        task = json.loads((root / "eval_finalists" / "child" / "2.json").read_text(encoding="utf-8"))
        self.assertEqual(task["global_task_index"], 1)
        self.assertEqual(task["runtime_artifact"]["orientation_items_sha256"], items_digest(["method"]))

    def test_heldout_requires_frozen_warmup_and_disables_learning(self):
        callback = self.make_callback()
        path = self.data()
        output = self.root / "terminal" / "task_logs"
        with self.assertRaises(ValueError):
            callback("webwalkerqa", "parent", [0], path, output)
        callback.heldout_warmup = []
        callback("webwalkerqa", "parent", [0, 1], path, output)
        self.assertTrue(all(not call[1]["enable_memory_evolution"] for call in self.processed))
        self.assertEqual(self.ingested, [])

    def test_wrong_answer_is_scoreable_and_judge_error_is_failure(self):
        callback = self.make_callback()
        path = self.data()
        original = callback.runner.process_item
        def wrong(*args, **kwargs):
            return {**original(*args, **kwargs), "judgement": "incorrect"}
        callback.runner.process_item = wrong
        callback("webwalkerqa", "parent", [0], path, self.root / "evolution" / "round_00" / "base_logs")
        self.assertEqual(callback.stage_events[0]["correct_count"], 0)
        callback.runner.process_item = lambda *args, **kwargs: {"status": "success", "judgement": "error"}
        with self.assertRaises(RuntimeError):
            callback("webwalkerqa", "parent", [0], path, self.root / "evolution" / "round_01" / "base_logs")


if __name__ == "__main__":
    unittest.main()
