#!/usr/bin/env python3
"""Submit a task to the Colab agent Git queue."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


QUEUE_BRANCH = "colab-agent-queue"
QUEUE_ROOT = ".colab-agent"


def run(args: list[str], cwd: Path | None = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=capture)


def git(cwd: Path, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=cwd, check=check, capture=capture)


def current_remote() -> str:
    return run(["git", "remote", "get-url", "origin"], capture=True).stdout.strip()


def branch_exists(remote: str, branch: str) -> bool:
    proc = run(["git", "ls-remote", "--heads", remote, branch], check=False, capture=True)
    return bool(proc.stdout.strip())


def ensure_queue_checkout(queue_dir: Path, remote: str, branch: str) -> None:
    if not queue_dir.exists():
        queue_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", remote, str(queue_dir)])
    git(queue_dir, "remote", "set-url", "origin", remote)
    if branch_exists(remote, branch):
        git(queue_dir, "fetch", "origin", branch)
        git(queue_dir, "checkout", "-B", branch, f"origin/{branch}")
        git(queue_dir, "reset", "--hard", f"origin/{branch}")
    else:
        git(queue_dir, "checkout", "--orphan", branch, check=False)
        for child in queue_dir.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for name in ("queued", "running", "done", "failed", "cancelled", "logs", "heartbeats", "results"):
            path = queue_dir / QUEUE_ROOT / name
            path.mkdir(parents=True, exist_ok=True)
            (path / ".gitkeep").write_text("")
        git(queue_dir, "add", QUEUE_ROOT)
        git(queue_dir, "commit", "-m", "Initialize Colab agent queue")
        git(queue_dir, "push", "-u", "origin", branch)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug[:80] or "task"


def task_id(name: str | None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(name or 'colab-task')}"


def parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def write_task(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "id": args.id or task_id(args.name),
        "name": args.name,
        "type": "shell",
        "status": "queued",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "cmd": args.cmd,
        "cwd": args.cwd,
        "timeout_seconds": args.timeout_seconds,
        "env": parse_env(args.env),
        "artifacts": args.artifact,
        "sync_repo": not args.no_sync_repo,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a shell task to the Colab agent queue.")
    parser.add_argument("--cmd", required=True, help="Shell command to run in Colab.")
    parser.add_argument("--name", help="Human-readable task name.")
    parser.add_argument("--id", help="Stable task id. Defaults to timestamp plus name.")
    parser.add_argument("--cwd", default="/content/masked-face-id")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--env", action="append", default=[], help="Environment variable as KEY=VALUE.")
    parser.add_argument("--artifact", action="append", default=[], help="Glob relative to cwd to copy into queue results.")
    parser.add_argument("--no-sync-repo", action="store_true", help="Do not reset repo cwd to origin/main before running.")
    parser.add_argument("--queue-dir", default=".agent_context/colab-agent-queue")
    parser.add_argument("--queue-branch", default=QUEUE_BRANCH)
    parser.add_argument("--remote", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task = write_task(args)
    print(json.dumps(task, indent=2, sort_keys=True))
    if args.dry_run:
        return

    queue_dir = Path(args.queue_dir).resolve()
    remote = args.remote or current_remote()
    ensure_queue_checkout(queue_dir, remote, args.queue_branch)
    task_path = queue_dir / QUEUE_ROOT / "queued" / f"{task['id']}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
    git(queue_dir, "add", str(task_path.relative_to(queue_dir)))
    git(queue_dir, "commit", "-m", f"Queue Colab task {task['id']}")
    git(queue_dir, "push", "origin", args.queue_branch)
    print(f"queued {task['id']} on {args.queue_branch}")


if __name__ == "__main__":
    main()
