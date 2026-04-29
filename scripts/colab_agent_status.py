#!/usr/bin/env python3
"""Inspect the Colab agent Git queue."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


QUEUE_BRANCH = "colab-agent-queue"
QUEUE_ROOT = ".colab-agent"


def run(args: list[str], cwd: Path | None = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=capture)


def git(cwd: Path, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=cwd, check=check, capture=capture)


def current_remote() -> str:
    return run(["git", "remote", "get-url", "origin"], capture=True).stdout.strip()


def ensure_checkout(queue_dir: Path, remote: str, branch: str) -> bool:
    if not queue_dir.exists():
        queue_dir.parent.mkdir(parents=True, exist_ok=True)
        clone = run(["git", "clone", "--branch", branch, "--single-branch", remote, str(queue_dir)], check=False)
        return clone.returncode == 0
    git(queue_dir, "fetch", "origin", branch, check=False)
    git(queue_dir, "checkout", "-B", branch, f"origin/{branch}", check=False)
    git(queue_dir, "reset", "--hard", f"origin/{branch}", check=False)
    return True


def load_tasks(queue_dir: Path) -> list[dict]:
    tasks = []
    root = queue_dir / QUEUE_ROOT
    for state in ("queued", "running", "done", "failed", "cancelled"):
        for path in sorted((root / state).glob("*.json")):
            data = json.loads(path.read_text())
            data["_state_dir"] = state
            tasks.append(data)
    return tasks


def tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Colab agent task status.")
    parser.add_argument("--queue-dir", default=".agent_context/colab-agent-queue")
    parser.add_argument("--queue-branch", default=QUEUE_BRANCH)
    parser.add_argument("--remote", default=None)
    parser.add_argument("--task-id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tail", type=int, default=0)
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir).resolve()
    remote = args.remote or current_remote()
    if not ensure_checkout(queue_dir, remote, args.queue_branch):
        raise SystemExit(f"Queue branch {args.queue_branch!r} does not exist yet.")
    tasks = load_tasks(queue_dir)
    if args.task_id:
        tasks = [task for task in tasks if task.get("id") == args.task_id]
    if args.json:
        print(json.dumps(tasks, indent=2, sort_keys=True))
    else:
        for task in tasks:
            print(
                f"{task.get('id')} [{task.get('status')}] "
                f"created={task.get('created_at')} returncode={task.get('returncode')} cmd={task.get('cmd')}"
            )
    if args.tail and args.task_id:
        root = queue_dir / QUEUE_ROOT
        print("\n--- stdout tail ---")
        print(tail(root / "logs" / f"{args.task_id}.stdout.log", args.tail))
        print("\n--- stderr tail ---")
        print(tail(root / "logs" / f"{args.task_id}.stderr.log", args.tail))


if __name__ == "__main__":
    main()
