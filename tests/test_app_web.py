"""The local HTTP workbench dispatches safely without real model calls."""

from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest

from philoagent_app.web import MAX_BODY, make_server


class FakeAgent:
    workspaces = {}
    calls = []

    def __init__(self, workspace, *, env_file=None):
        self.workspace = Path(workspace)

    @classmethod
    def create_from_text(cls, workspace, filename, content):
        workspace = Path(workspace)
        if workspace in cls.workspaces:
            raise FileExistsError("choose a new directory")
        workspace.mkdir(parents=True)
        cls.workspaces[workspace] = {"initialized": True, "system": {"filename": filename, "items": [content]}, "tasks": [], "busy": False}
        cls.calls.append(("init", filename, content))
        return cls(workspace)

    def info(self):
        return self.workspaces.get(self.workspace, {"initialized": False, "tasks": [], "busy": False})

    def run(self, task, request_id=None):
        self.calls.append(("run", task, request_id))
        return {"status": "completed", "record": {"answer": "离线作品", "grade": {"score": 81.5}, "agent_trajectory": ["search"]}}

    def evolve(self, tasks, rounds=1, request_id=None):
        self.calls.append(("evolve", tasks, rounds, request_id))
        return {"status": "completed", "operation": "evolve"}


class AppWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        FakeAgent.workspaces, FakeAgent.calls = {}, []
        self.server = make_server(self.root, port=0, agent_class=FakeAgent)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        result = response.read()
        metadata = dict(response.getheaders())
        connection.close()
        return response.status, result, metadata

    def test_refresh_is_read_only_and_static_assets_have_security_headers(self):
        for path in ("/", "/style.css", "/app.js", "/api/info", "/api/config"):
            status, body, headers = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")
            self.assertTrue(body)
        self.assertEqual(FakeAgent.calls, [])
        self.assertFalse((self.root / "default").exists())

    def test_import_task_and_evolution_workflow(self):
        source = "完整原文\n不要摘要"
        self.assertEqual(self.request("POST", "/api/init", {"name": "test-agent", "filename": "system.md", "content": source})[0], 200)
        info = json.loads(self.request("GET", "/api/info?workspace=test-agent")[1])
        self.assertEqual(info["system"]["items"], [source])
        self.assertEqual(self.request("POST", "/api/init", {"name": "test-agent", "filename": "system.md", "content": "new"})[0], 409)
        task = {"domain": "business", "question": "比较市场结构"}
        result = self.request("POST", "/api/run", {"workspace": "test-agent", "task": task, "request_id": "r1"})
        self.assertEqual(json.loads(result[1])["record"]["answer"], "离线作品")
        self.assertEqual(FakeAgent.calls[-1], ("run", task, "r1"))
        tasks = [{**task, "task_id": str(index)} for index in range(4)]
        self.assertEqual(self.request("POST", "/api/evolve", {"workspace": "test-agent", "tasks": tasks, "rounds": 1, "request_id": "e1"})[0], 200)
        self.assertEqual(FakeAgent.calls[-1], ("evolve", tasks, 1, "e1"))

    def test_host_origin_path_and_content_constraints(self):
        self.assertEqual(self.request("GET", "/api/info", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/run", {}, headers={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.request("GET", "/api/info?workspace=../outside")[0], 400)
        self.assertEqual(self.request("GET", "/../agent.json")[0], 404)
        self.assertEqual(self.request("POST", "/api/run", {}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/init", {"name": "safe", "filename": "../system.md", "content": "x"})[0], 400)
        self.assertEqual(self.request("POST", "/api/run", {"workspace": "default", "task": {}})[0], 400)
        self.assertEqual(FakeAgent.calls, [])

    def test_oversize_body_rejected_before_dispatch(self):
        # Reject on headers before accepting a body. Sending a large unread body
        # while the server closes can legitimately reset a Windows socket.
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.putrequest("POST", "/api/init")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(MAX_BODY + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        connection.close()
        self.assertEqual(FakeAgent.calls, [])

    def test_nonlocal_binding_rejected(self):
        with self.assertRaises(ValueError):
            make_server(self.root, host="0.0.0.0", agent_class=FakeAgent)

    def test_model_and_user_text_are_never_assigned_to_html(self):
        script = (Path(__file__).parents[1] / "philoagent_app" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("eval(", script)
        self.assertIn("textContent", script)


if __name__ == "__main__":
    unittest.main()
