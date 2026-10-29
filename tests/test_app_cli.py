"""CLI routes remain offline and call the one public Agent implementation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from philoagent_app import cli


class AppCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.agent = MagicMock()
        self.agent.info.return_value = {"initialized": True}
        self.agent.run.return_value = {"status": "completed", "record": {"answer": "离线作品"}}
        self.agent.evolve.return_value = {"status": "completed"}
        self.agent.evaluate.return_value = {"status": "completed"}
        self.agent_class = MagicMock(return_value=self.agent)
        self.agent_class.create.return_value = self.agent
        self.patcher = patch.object(cli, "Agent", self.agent_class)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def invoke(self, args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = cli.main(args)
        return code, output.getvalue(), error.getvalue()

    def test_init_and_options_on_either_side_of_command(self):
        workspace, source = self.root / "agent", self.root / "system.md"
        for args in (["--workspace", str(workspace), "init", "--system", str(source)],
                     ["init", "--workspace", str(workspace), "--system", str(source)]):
            code, output, _ = self.invoke(args)
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output)["initialized"])
            self.agent_class.create.assert_called_with(workspace, source)

    def test_run_passes_materials_and_requirements_without_rewriting(self):
        code, output, _ = self.invoke(["run", "--domain", "social", "--question", "研究平台治理",
                                      "--materials", "完整材料\n第二行", "--deliverable", "研究备忘录",
                                      "--criterion", "交代反例", "--request-id", "one"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)["record"]["answer"], "离线作品")
        self.agent.run.assert_called_once_with({"domain": "social", "question": "研究平台治理",
                                              "materials": ["完整材料\n第二行"], "deliverable": "研究备忘录",
                                              "task_criteria": ["交代反例"]}, request_id="one")

    def test_json_task_file_and_env_are_read_only_inputs(self):
        source = self.root / "task.json"
        task = {"domain": "humanities", "question": "文本解释", "task_id": "h1"}
        source.write_text(json.dumps(task), encoding="utf-8-sig")
        env = self.root / "private.env"
        code, _, _ = self.invoke(["run", "--task-file", str(source), "--env-file", str(env)])
        self.assertEqual(code, 0)
        self.agent.run.assert_called_once_with(task, request_id=None)
        self.assertEqual(self.agent_class.call_args.kwargs["env_file"], env)
        self.assertFalse(env.exists())

    def test_evolve_and_evaluate_dispatch_to_core(self):
        tasks = self.root / "tasks.jsonl"
        self.assertEqual(self.invoke(["evolve", "--tasks", str(tasks), "--rounds", "2"])[0], 0)
        self.agent.evolve.assert_called_once_with(tasks, rounds=2, request_id=None)
        self.assertEqual(self.invoke(["evaluate", "--tasks", str(tasks), "--current-only"])[0], 0)
        self.agent.evaluate.assert_called_once_with(tasks, compare_initial=False, request_id=None)

    def test_incomplete_evaluation_is_reported_as_nonzero(self):
        self.agent.evaluate.return_value = {"status": "incomplete", "missing_scores": 1}
        self.assertEqual(self.invoke(["evaluate", "--tasks", "evaluation.jsonl"])[0], 1)

    def test_failure_and_interrupt_have_nonzero_exit_without_retry(self):
        self.agent.run.return_value = {"status": "failed", "record": {"answer": "已保存"}}
        self.assertEqual(self.invoke(["run", "--domain", "business", "--question", "任务"])[0], 1)
        self.agent.run.side_effect = KeyboardInterrupt
        code, _, error = self.invoke(["run", "--domain", "business", "--question", "任务"])
        self.assertEqual(code, 130)
        self.assertIn("不会自动重试", error)
        self.assertEqual(self.agent.run.call_count, 2)

    def test_serve_uses_workspace_parent_and_loopback(self):
        workspace = self.root / "a"
        with patch("philoagent_app.web.serve") as serve:
            self.assertEqual(self.invoke(["serve", "--workspace", str(workspace), "--port", "9001"])[0], 0)
        serve.assert_called_once_with(workspace_root=self.root, initial_workspace="a", host="127.0.0.1", port=9001, env_file=None)


if __name__ == "__main__":
    unittest.main()
