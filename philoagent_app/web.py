"""A loopback-only, dependency-free local workbench.

Requests perform the same synchronous Agent operations as the CLI. A threaded
server permits read-only polling while the Agent's workspace lock serializes
mutations. No model calls happen on GET or on page refresh.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import Agent


STATIC = Path(__file__).with_name("static")
MAX_BODY = 2 * 1024 * 1024
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def make_server(workspace_root: str | Path, *, initial_workspace: str = "default",
                host: str = "127.0.0.1", port: int = 8765,
                env_file: str | Path | None = None, agent_class: type = Agent) -> ThreadingHTTPServer:
    """Create, but do not start, a local server (also useful for offline tests)."""
    if host not in LOCAL_HOSTS:
        raise ValueError("工作台仅允许监听本机 localhost / 127.0.0.1 / ::1")
    if not WORKSPACE_NAME.fullmatch(initial_workspace):
        raise ValueError("工作目录名称只接受英文、数字、连字符和下划线")
    root = Path(workspace_root).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "PhiloAgentLocal/1"

        def log_message(self, format: str, *args: Any) -> None:
            # Tasks and imported documents must not end up in HTTP logs.
            return

        def _reply(self, status: int, value: Any, content_type: str = "application/json; charset=utf-8") -> None:
            payload = (json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
                       if content_type.startswith("application/json") else value)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Closing the page must not request a retry or cancel work.

        def _allowed(self) -> bool:
            host_header = self.headers.get("Host", "")
            try:
                parsed = urlsplit("http://" + host_header)
                host_ok = (parsed.hostname in LOCAL_HOSTS and parsed.port == self.server.server_port
                           and not parsed.username and not parsed.password
                           and not parsed.path and not parsed.query and not parsed.fragment)
            except ValueError:
                host_ok = False
            if not host_ok:
                self._reply(HTTPStatus.FORBIDDEN, {"error": "仅接受本机地址的 Host"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host_header:
                self._reply(HTTPStatus.FORBIDDEN, {"error": "拒绝跨来源请求"})
                return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._reply(HTTPStatus.FORBIDDEN, {"error": "拒绝跨站请求"})
                return False
            return True

        def _workspace(self, name: Any) -> Path:
            if not isinstance(name, str) or not WORKSPACE_NAME.fullmatch(name):
                raise ValueError("智能体名称需为 1–64 位英文、数字、连字符或下划线")
            destination = (root / name).resolve()
            if destination.parent != root:
                raise ValueError("工作目录必须位于智能体存放目录内")
            return destination

        def _agent(self, name: Any) -> Any:
            return agent_class(self._workspace(name), env_file=env_file)

        def do_GET(self) -> None:
            if not self._allowed():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/info":
                    name = parse_qs(parsed.query).get("workspace", [initial_workspace])[0]
                    self._reply(HTTPStatus.OK, self._agent(name).info())
                elif parsed.path == "/api/config":
                    names = sorted(item.name for item in root.iterdir()
                                   if item.is_dir() and WORKSPACE_NAME.fullmatch(item.name)
                                   and (item.resolve().parent == root)) if root.exists() else []
                    self._reply(HTTPStatus.OK, {"initial_workspace": initial_workspace, "workspaces": names})
                elif parsed.path in {"/", "/index.html", "/style.css", "/app.js"}:
                    filename = "index.html" if parsed.path in {"/", "/index.html"} else parsed.path[1:]
                    mime = {"index.html": "text/html; charset=utf-8", "style.css": "text/css; charset=utf-8",
                            "app.js": "text/javascript; charset=utf-8"}[filename]
                    self._reply(HTTPStatus.OK, (STATIC / filename).read_bytes(), mime)
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:
            if not self._allowed():
                return
            try:
                content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type != "application/json":
                    self._reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "仅接受 application/json"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_BODY:
                    self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求内容不得超过 2 MiB"})
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("请求必须为 JSON 对象")
                path = urlsplit(self.path).path
                if path == "/api/init":
                    workspace = self._workspace(body.get("name"))
                    filename, content = body.get("filename"), body.get("content")
                    if not isinstance(filename, str) or not isinstance(content, str):
                        raise ValueError("需提供文件名与文本内容")
                    if Path(filename).name != filename or Path(filename).suffix.lower() not in {".json", ".md", ".txt"}:
                        raise ValueError("请选择 JSON / Markdown / TXT 文件")
                    result = agent_class.create_from_text(workspace, filename, content).info()
                elif path in {"/api/run", "/api/evolve"}:
                    agent = self._agent(body.get("workspace"))
                    request_id = body.get("request_id")
                    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
                        raise ValueError("每次执行需提供 request_id")
                    if path == "/api/run":
                        task = body.get("task")
                        if not isinstance(task, dict):
                            raise ValueError("需提供任务对象")
                        result = agent.run(task, request_id=request_id)
                    else:
                        tasks = body.get("tasks")
                        rounds = body.get("rounds", 1)
                        if not isinstance(tasks, list):
                            raise ValueError("开发任务需为任务列表")
                        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
                            raise ValueError("演化轮数必须为正整数")
                        result = agent.evolve(tasks, rounds=rounds, request_id=request_id)
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
                    return
                self._reply(HTTPStatus.OK, result)
            except Exception as exc:
                self._error(exc)

        def _error(self, exc: Exception) -> None:
            busy = "busy" in type(exc).__name__.lower() or "lock" in type(exc).__name__.lower()
            status = (HTTPStatus.CONFLICT if busy or isinstance(exc, FileExistsError)
                      else HTTPStatus.BAD_REQUEST if isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError))
                      else HTTPStatus.INTERNAL_SERVER_ERROR)
            self._reply(status, {"error": type(exc).__name__, "message": str(exc)})

    server_class = ThreadingHTTPServer
    if host == "::1":
        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        server_class = IPv6Server
    server = server_class((host, port), Handler)
    server.daemon_threads = False
    return server


def serve(workspace_root: str | Path, *, initial_workspace: str = "default",
          host: str = "127.0.0.1", port: int = 8765,
          env_file: str | Path | None = None) -> None:
    server = make_server(workspace_root, initial_workspace=initial_workspace,
                         host=host, port=port, env_file=env_file)
    url_host = "[::1]" if host == "::1" else host
    print(f"PhiloAgent 工作台：http://{url_host}:{server.server_port}")
    print(f"智能体目录：{Path(workspace_root).resolve()}\n运行任务会使用已配置的模型并产生实际调用费用。")
    try:
        server.serve_forever()
    finally:
        server.server_close()
