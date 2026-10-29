"""Minimal local command line for the persistent PhiloAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from . import Agent


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "agent_data" / "default"
DOMAINS = ("humanities", "social", "business")


def _common(parser: argparse.ArgumentParser, *, subordinate: bool = False) -> None:
    default = argparse.SUPPRESS if subordinate else DEFAULT_WORKSPACE
    parser.add_argument("--workspace", type=Path, default=default, help="智能体的独立工作目录")
    parser.add_argument("--env-file", type=Path, default=argparse.SUPPRESS if subordinate else None,
                        help="只读加载模型配置文件；不修改原实验配置")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m philoagent_app", description="PhiloAgent · 研究与写作智能体")
    _common(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="从 JSON / Markdown / TXT 哲学系统新建智能体")
    _common(init, subordinate=True)
    init.add_argument("--system", required=True, type=Path)
    run = commands.add_parser("run", help="执行任务、独立评分并保存经验")
    _common(run, subordinate=True)
    run.add_argument("--domain", choices=DOMAINS)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="研究或写作任务")
    source.add_argument("--task-file", type=Path, help="单个任务 JSON 文件")
    run.add_argument("--materials", action="append", default=[], help="可重复指定的补充材料原文")
    run.add_argument("--deliverable", help="交付要求")
    run.add_argument("--criterion", action="append", default=[], help="可重复指定的任务评判标准")
    run.add_argument("--request-id", help="相同请求标识不会自动重复执行")
    evolve = commands.add_parser("evolve", help="用独立开发题集手动演化")
    _common(evolve, subordinate=True)
    evolve.add_argument("--tasks", type=Path, required=True, help="至少四题的开发任务 JSONL")
    evolve.add_argument("--rounds", type=int, default=1)
    evolve.add_argument("--request-id")
    evaluate = commands.add_parser("evaluate", help="隔离评测初始化状态和当前状态")
    _common(evaluate, subordinate=True)
    evaluate.add_argument("--tasks", type=Path, required=True, help="独立留出的评测任务 JSONL")
    evaluate.add_argument("--current-only", action="store_true", help="仅评测当前状态")
    evaluate.add_argument("--request-id")
    serve = commands.add_parser("serve", help="启动本机单页工作台")
    _common(serve, subordinate=True)
    serve.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--root", type=Path, help="网页智能体存放目录；默认为 --workspace 的父目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            result = Agent.create(args.workspace, args.system).info()
        elif args.command == "serve":
            from .web import serve
            serve(workspace_root=args.root or args.workspace.parent,
                  initial_workspace=args.workspace.name, host=args.host, port=args.port,
                  env_file=args.env_file)
            return 0
        else:
            agent = Agent(args.workspace, env_file=args.env_file)
            if args.command == "run":
                if args.task_file:
                    task = json.loads(args.task_file.read_text(encoding="utf-8-sig"))
                    if not isinstance(task, dict):
                        raise ValueError("--task-file 必须包含一个任务 JSON 对象")
                else:
                    if not args.domain:
                        parser.error("使用 --question 时需要 --domain")
                    task = {"domain": args.domain, "question": args.question, "materials": args.materials}
                    if args.deliverable:
                        task["deliverable"] = args.deliverable
                    if args.criterion:
                        task["task_criteria"] = args.criterion
                result = agent.run(task, request_id=args.request_id)
            elif args.command == "evolve":
                result = agent.evolve(args.tasks, rounds=args.rounds, request_id=args.request_id)
            else:
                result = agent.evaluate(args.tasks, compare_initial=not args.current_only, request_id=args.request_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 1 if result.get("status") in {"failed", "error", "interrupted", "incomplete"} else 0
    except KeyboardInterrupt:
        print("已中断。已保存的作品保留；不会自动重试。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
