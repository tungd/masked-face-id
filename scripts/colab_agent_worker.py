#!/usr/bin/env python3
"""Colab-side worker for a small Git-backed task queue.

MCP only needs to start this script once inside a Colab notebook. After that,
the worker polls a queue branch, runs submitted shell tasks in the Colab
runtime, and pushes status/results back to the queue branch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


QUEUE_BRANCH = "colab-agent-queue"
QUEUE_ROOT = ".colab-agent"
STATES = ("queued", "running", "done", "failed", "cancelled")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def run(
    args: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )


def git(cwd: Path, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=cwd, check=check, capture=capture)


def remote_url(repo_dir: Path) -> str:
    env_remote = os.environ.get("COLAB_AGENT_REMOTE")
    if env_remote:
        return env_remote
    proc = git(repo_dir, "remote", "get-url", "origin", capture=True)
    return proc.stdout.strip()


def token_from_colab_secret() -> str | None:
    for name in ("COLAB_AGENT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore

        for name in ("COLAB_AGENT_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
            value = userdata.get(name)
            if value:
                return value
    except Exception:
        return None
    return None


def authed_remote(url: str, token: str | None) -> str:
    if not token or not url.startswith("https://github.com/"):
        return url
    return "https://x-access-token:" + token + "@" + url.removeprefix("https://")


def branch_exists(remote: str, branch: str) -> bool:
    proc = run(["git", "ls-remote", "--heads", remote, branch], check=False, capture=True)
    return bool(proc.stdout.strip())


def ensure_queue_checkout(repo_dir: Path, queue_dir: Path, branch: str) -> None:
    remote = authed_remote(remote_url(repo_dir), token_from_colab_secret())
    if not queue_dir.exists():
        queue_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", remote, str(queue_dir)], check=True)
    git(queue_dir, "remote", "set-url", "origin", remote)
    if branch_exists(remote, branch):
        git(queue_dir, "fetch", "origin", branch)
        git(queue_dir, "checkout", "-B", branch, f"origin/{branch}")
    else:
        git(queue_dir, "checkout", "--orphan", branch, check=False)
        for child in queue_dir.iterdir():
            if child.name != ".git":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        root = queue_dir / QUEUE_ROOT
        for state in STATES:
            (root / state).mkdir(parents=True, exist_ok=True)
            (root / state / ".gitkeep").write_text("")
        for name in ("logs", "heartbeats", "results"):
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / ".gitkeep").write_text("")
        git(queue_dir, "add", QUEUE_ROOT)
        git(queue_dir, "commit", "-m", "Initialize Colab agent queue")
        git(queue_dir, "push", "-u", "origin", branch)


def refresh_queue(queue_dir: Path, branch: str) -> bool:
    fetch = git(queue_dir, "fetch", "origin", branch, check=False, capture=True)
    if fetch.returncode != 0:
        print("queue fetch failed:", fetch.stderr[-1000:], flush=True)
        return False
    git(queue_dir, "checkout", "-B", branch, f"origin/{branch}", check=False)
    git(queue_dir, "reset", "--hard", f"origin/{branch}", check=False)
    return True


def ensure_queue_dirs(queue_dir: Path) -> None:
    root = queue_dir / QUEUE_ROOT
    for name in (*STATES, "logs", "heartbeats", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def commit_push(queue_dir: Path, message: str, branch: str) -> bool:
    git(queue_dir, "add", QUEUE_ROOT, check=True)
    diff = git(queue_dir, "diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        return True
    commit = git(queue_dir, "commit", "-m", message, check=False, capture=True)
    if commit.returncode != 0:
        print("queue commit failed:", commit.stderr[-1000:], flush=True)
        return False
    push = git(queue_dir, "push", "origin", branch, check=False, capture=True)
    if push.returncode != 0:
        print("queue push failed:", push.stderr[-1000:], flush=True)
        return False
    return True


def queued_tasks(queue_dir: Path) -> list[Path]:
    return sorted((queue_dir / QUEUE_ROOT / "queued").glob("*.json"))


def claim_task(queue_dir: Path, task_path: Path, branch: str, worker_id: str) -> tuple[dict[str, Any], Path] | None:
    task = load_json(task_path)
    task_id = task.get("id") or task_path.stem
    task["id"] = task_id
    task["status"] = "running"
    task["worker_id"] = worker_id
    task["claimed_at"] = now()
    task["started_at"] = task["claimed_at"]
    running_path = queue_dir / QUEUE_ROOT / "running" / f"{task_id}.json"
    task_path.unlink()
    write_json(running_path, task)
    if commit_push(queue_dir, f"Claim Colab task {task_id}", branch):
        return task, running_path
    refresh_queue(queue_dir, branch)
    return None


def update_heartbeat(queue_dir: Path, task: dict[str, Any], branch: str, push: bool) -> None:
    heartbeat = {
        "id": task["id"],
        "status": "running",
        "worker_id": task.get("worker_id"),
        "updated_at": now(),
    }
    write_json(queue_dir / QUEUE_ROOT / "heartbeats" / f"{task['id']}.json", heartbeat)
    if push:
        commit_push(queue_dir, f"Heartbeat Colab task {task['id']}", branch)


def copy_artifacts(task: dict[str, Any], queue_dir: Path, cwd: Path, max_bytes: int) -> list[str]:
    copied: list[str] = []
    result_dir = queue_dir / QUEUE_ROOT / "results" / task["id"]
    result_dir.mkdir(parents=True, exist_ok=True)
    for pattern in task.get("artifacts", []):
        for item in glob.glob(str(cwd / pattern), recursive=True):
            path = Path(item)
            if not path.is_file():
                continue
            if path.stat().st_size > max_bytes:
                continue
            dest = result_dir / path.name
            shutil.copy2(path, dest)
            copied.append(str(dest.relative_to(queue_dir / QUEUE_ROOT)))
    return copied


def run_task(
    task: dict[str, Any],
    queue_dir: Path,
    running_path: Path,
    branch: str,
    repo_dir: Path,
    heartbeat_seconds: int,
    push_heartbeats: bool,
    max_artifact_bytes: int,
) -> None:
    task_id = task["id"]
    cwd = Path(task.get("cwd") or repo_dir)
    if not cwd.is_absolute():
        cwd = repo_dir / cwd
    cwd.mkdir(parents=True, exist_ok=True)

    if task.get("sync_repo", True) and cwd == repo_dir:
        git(repo_dir, "fetch", "origin", "main", check=False)
        git(repo_dir, "reset", "--hard", "origin/main", check=False)

    stdout_path = queue_dir / QUEUE_ROOT / "logs" / f"{task_id}.stdout.log"
    stderr_path = queue_dir / QUEUE_ROOT / "logs" / f"{task_id}.stderr.log"
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in task.get("env", {}).items()})
    timeout = int(task.get("timeout_seconds") or 0)
    cmd = task["cmd"]
    started = time.monotonic()
    returncode = 124
    error = None

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=stdout, stderr=stderr, env=env)
        last_heartbeat = 0.0
        while True:
            rc = process.poll()
            elapsed = time.monotonic() - started
            if elapsed - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = elapsed
                task["last_heartbeat_at"] = now()
                write_json(running_path, task)
                update_heartbeat(queue_dir, task, branch, push_heartbeats)
            if rc is not None:
                returncode = rc
                break
            if timeout and elapsed > timeout:
                process.kill()
                error = f"timeout after {timeout} seconds"
                returncode = 124
                break
            time.sleep(1)

    finished_at = now()
    task["finished_at"] = finished_at
    task["returncode"] = returncode
    task["duration_seconds"] = round(time.monotonic() - started, 3)
    if error:
        task["error"] = error
    task["logs"] = {
        "stdout": str(stdout_path.relative_to(queue_dir / QUEUE_ROOT)),
        "stderr": str(stderr_path.relative_to(queue_dir / QUEUE_ROOT)),
    }
    task["artifacts_copied"] = copy_artifacts(task, queue_dir, cwd, max_artifact_bytes)
    task["status"] = "done" if returncode == 0 else "failed"

    final_dir = queue_dir / QUEUE_ROOT / task["status"]
    final_path = final_dir / f"{task_id}.json"
    if running_path.exists():
        running_path.unlink()
    write_json(final_path, task)
    commit_push(queue_dir, f"Finish Colab task {task_id}: {task['status']}", branch)


def worker_loop(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    queue_dir = Path(args.queue_dir).resolve()
    worker_id = args.worker_id or f"colab-{socket.gethostname()}-{os.getpid()}"
    ensure_queue_checkout(repo_dir, queue_dir, args.queue_branch)
    ensure_queue_dirs(queue_dir)
    print(
        json.dumps(
            {
                "event": "worker_started",
                "worker_id": worker_id,
                "repo_dir": str(repo_dir),
                "queue_dir": str(queue_dir),
                "queue_branch": args.queue_branch,
            }
        ),
        flush=True,
    )
    while True:
        refresh_queue(queue_dir, args.queue_branch)
        ensure_queue_dirs(queue_dir)
        tasks = queued_tasks(queue_dir)
        if not tasks:
            time.sleep(args.poll_seconds)
            continue
        claimed = claim_task(queue_dir, tasks[0], args.queue_branch, worker_id)
        if not claimed:
            time.sleep(args.poll_seconds)
            continue
        task, running_path = claimed
        print(json.dumps({"event": "task_claimed", "id": task["id"], "cmd": task["cmd"]}), flush=True)
        run_task(
            task,
            queue_dir,
            running_path,
            args.queue_branch,
            repo_dir,
            args.heartbeat_seconds,
            args.push_heartbeats,
            args.max_artifact_bytes,
        )
        if args.once:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Colab Git-backed task worker.")
    parser.add_argument("--repo-dir", default="/content/masked-face-id")
    parser.add_argument("--queue-dir", default="/content/colab-agent-queue")
    parser.add_argument("--queue-branch", default=QUEUE_BRANCH)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--heartbeat-seconds", type=int, default=120)
    parser.add_argument("--push-heartbeats", action="store_true")
    parser.add_argument("--max-artifact-bytes", type=int, default=10_000_000)
    parser.add_argument("--worker-id")
    parser.add_argument("--once", action="store_true", help="Process one queued task, then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    worker_loop(parse_args())
