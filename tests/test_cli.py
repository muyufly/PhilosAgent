"""Offline launcher checks using a tiny fake upstream checkout."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from philoagent import cli
from philoagent.orientation import load


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.upstream = self.project / "vendor" / "MemEvolve"
        self.flash = self.upstream / "Flash-Searcher-main"
        self.flash.mkdir(parents=True)
        for filename in ("evolve_cli.py", "run_flash_searcher_webwalkerqa.py"):
            (self.flash / filename).write_text("raise RuntimeError('test must never execute upstream')\n", encoding="utf-8")
        for filename in (".env", ".env.local"):
            (self.flash / filename).write_text("OPENAI_API_KEY=upstream-secret", encoding="utf-8")
        for directory in ("data", "storage", "__pycache__", ".git"):
            folder = self.flash / directory
            folder.mkdir()
            (folder / "do-not-copy.txt").write_text("heldout-or-local-state", encoding="utf-8")
        (self.project / "philoagent").mkdir()
        (self.project / "philoagent" / "adapter.py").write_text("VERSION = 1\n", encoding="utf-8")
        (self.project / "seeds").mkdir()
        for seed in cli.SEEDS:
            (self.project / "seeds" / f"{seed}.json").write_text(json.dumps({"items": [f"{seed} initial method"]}), encoding="utf-8")
        self.source = self.root / "dev.jsonl"
        self.source.write_text("\n".join(json.dumps({"id": str(i), "question": f"Q{i}", "answer": f"A{i}", "root_url": f"https://site{i}.example/"}) for i in range(12)) + "\n", encoding="utf-8")
        for name, value in (("PROJECT", self.project), ("UPSTREAM", self.upstream), ("FLASH", self.flash)):
            patcher = patch.object(cli, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env_patcher = patch.dict(os.environ, {}, clear=True)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    def create_run(self, name="sample", seed="engineering"):
        return Path(cli.init_run(name, seed, self.source)["run_dir"])

    def test_init_isolates_code_orientation_and_development_data(self):
        first = self.create_run("first")
        second = self.create_run("second", "dialectical")
        self.assertEqual(load(first / "lineage", cli.PROVIDER).items, ("engineering initial method",))
        self.assertEqual(load(second / "lineage", cli.PROVIDER).items, ("dialectical initial method",))
        copied = first / "workspace" / "data" / "webwalkerqa" / "webwalkerqa_subset_170.jsonl"
        self.assertEqual(copied.read_bytes(), self.source.read_bytes())
        for forbidden in (".env", ".env.local", "storage", "__pycache__", ".git", "data/do-not-copy.txt"):
            self.assertFalse((first / "workspace" / forbidden).exists(), forbidden)
        (first / "workspace" / "evolve_cli.py").write_text("CHANGED = True\n", encoding="utf-8")
        self.assertEqual((second / "workspace" / "evolve_cli.py").read_bytes(), (self.flash / "evolve_cli.py").read_bytes())
        before = (first / "run.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.create_run("first")
        self.assertEqual((first / "run.json").read_bytes(), before)

    def test_dry_run_prints_command_without_execution_or_credentials(self):
        run = self.create_run()
        secrets = {"OPENAI_API_KEY": "test-private-openai", "SERPER_API_KEY": "test-private-serper", "JINA_API_KEY": "test-private-jina", "DEFAULT_MODEL": "offline-test-model"}
        output = io.StringIO()
        with patch.dict(os.environ, secrets), patch.object(cli.subprocess, "run", side_effect=AssertionError("dry-run executed a child process")) as execute, contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["run", str(run), "--tasks", "2", "--dry-run"]), 0)
        execute.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertFalse(report["executed"])
        self.assertEqual(report["cwd"], str(run / "workspace"))
        self.assertIn("run_flash_searcher_webwalkerqa.py", report["command"])
        self.assertEqual(report["missing_configuration"], [])
        for secret in (secrets["OPENAI_API_KEY"], secrets["SERPER_API_KEY"], secrets["JINA_API_KEY"]):
            self.assertNotIn(secret, output.getvalue())
        self.assertFalse((run / "task_results.jsonl").exists())

    def test_modified_development_data_is_rejected_before_launch(self):
        run = self.create_run()
        copied = run / "workspace" / "data" / "webwalkerqa" / "webwalkerqa_subset_170.jsonl"
        copied.write_bytes(copied.read_bytes() + b"\n")
        with patch.object(cli.subprocess, "run") as execute, self.assertRaisesRegex(ValueError, "dataset differs"):
            cli.launch(run, dry_run=True)
        execute.assert_not_called()

    def test_endpoint_aliases_and_configuration_requirements(self):
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "https://endpoint.example/v1"}, clear=True):
                values = cli.environment()
                self.assertEqual(values["OPENAI_BASE_URL"], values["OPENAI_API_BASE"])
                self.assertEqual(values[name], "https://endpoint.example/v1")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://a.example", "OPENAI_API_BASE": "https://b.example"}, clear=True):
            with self.assertRaisesRegex(ValueError, "same endpoint"):
                cli.environment()
        configured = {"DEFAULT_MODEL": "model", "OPENAI_API_KEY": "key", "SERPER_API_KEY": "key", "WEB_ACCESS_PROVIDER": "crawl4ai"}
        self.assertEqual(cli.missing_configuration(configured), [])
        self.assertEqual(set(cli.missing_configuration(configured, evolving=True)), {"ANALYSIS_MODEL", "GENERATION_MODEL"})

    def test_invalid_env_reports_line_without_echoing_value(self):
        path = self.project / ".env"
        path.write_text("not valid env test-secret\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"lines \[1\]") as caught:
            cli.environment()
        self.assertNotIn("test-secret", str(caught.exception))

    def test_resume_uses_remaining_rounds_and_current_best_provider(self):
        run = self.create_run()
        state_path = run / "evolution" / "evolve_state.json"
        state_path.parent.mkdir()
        state = {"round": 2, "dataset_cursor": 8, "best_provider": "evolved_candidate"}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with patch.object(cli.subprocess, "run", side_effect=AssertionError("offline test executed upstream")) as execute:
            evolution = cli.launch(run, mode="evolve", rounds=3, batch_size=2, dry_run=True)
            command = evolution["command"]
            self.assertEqual(command[command.index("--num-rounds") + 1], "1")
            task = cli.launch(run, mode="task", dry_run=True)
            self.assertEqual(task["command"][task["command"].index("--memory_provider") + 1], "evolved_candidate")
            complete = cli.launch(run, mode="evolve", rounds=2, dry_run=True)
            self.assertFalse(complete["executed"])
            self.assertEqual(complete["completed_rounds"], 2)
            self.assertNotIn("command", complete)
            state["dataset_cursor"] = 10
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Insufficient development tasks"):
                cli.launch(run, mode="evolve", rounds=3, batch_size=2, dry_run=True)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
