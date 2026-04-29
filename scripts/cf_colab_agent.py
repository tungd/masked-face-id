#!/usr/bin/env python3
"""Cloudflare Queues backed Colab task agent.

This avoids storing a GitHub token in Colab. The Colab runtime only needs a
Cloudflare API token with Queues read/write permission for a task queue and a
result queue.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API_BASE = "https://api.cloudflare.com/client/v4"
MAX_EVENT_BODY_CHARS = 95_000


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def env_or_secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return None


def require_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    account_id = args.account_id or env_or_secret("CF_ACCOUNT_ID")
    task_queue_id = args.task_queue_id or env_or_secret("CF_TASK_QUEUE_ID")
    result_queue_id = args.result_queue_id or env_or_secret("CF_RESULT_QUEUE_ID")
    token = args.token or env_or_secret("CF_QUEUES_TOKEN") or env_or_secret("CLOUDFLARE_API_TOKEN")
    missing = [
        name
        for name, value in [
            ("CF_ACCOUNT_ID", account_id),
            ("CF_TASK_QUEUE_ID", task_queue_id),
            ("CF_RESULT_QUEUE_ID", result_queue_id),
            ("CF_QUEUES_TOKEN", token),
        ]
        if not value
    ]
    if missing:
        raise SystemExit("Missing Cloudflare queue config: " + ", ".join(missing))
    return str(account_id), str(task_queue_id), str(result_queue_id), str(token)


def cf_request(
    method: str,
    path: str,
    token: str,
    body: dict[str, Any] | None = None,
    api_base: str = DEFAULT_API_BASE,
) -> dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        api_base.rstrip("/") + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cloudflare API {method} {path} failed: {exc.code} {detail}") from exc
    parsed = json.loads(payload)
    if not parsed.get("success", False):
        raise RuntimeError(f"Cloudflare API {method} {path} failed: {parsed}")
    return parsed


def queue_path(account_id: str, queue_id: str, suffix: str) -> str:
    return f"/accounts/{account_id}/queues/{queue_id}/messages{suffix}"


def push_message(account_id: str, queue_id: str, token: str, message: dict[str, Any]) -> None:
    cf_request(
        "POST",
        queue_path(account_id, queue_id, ""),
        token,
        {"body": json.dumps(message), "content_type": "text"},
    )


def pull_messages(
    account_id: str,
    queue_id: str,
    token: str,
    batch_size: int,
    visibility_timeout_ms: int,
) -> list[dict[str, Any]]:
    response = cf_request(
        "POST",
        queue_path(account_id, queue_id, "/pull"),
        token,
        {"batch_size": batch_size, "visibility_timeout_ms": visibility_timeout_ms},
    )
    return response.get("result", {}).get("messages", [])


def ack_messages(account_id: str, queue_id: str, token: str, messages: list[dict[str, Any]]) -> None:
    leases = [{"lease_id": msg["lease_id"]} for msg in messages if msg.get("lease_id")]
    if leases:
        cf_request("POST", queue_path(account_id, queue_id, "/ack"), token, {"acks": leases, "retries": []})


def retry_messages(account_id: str, queue_id: str, token: str, messages: list[dict[str, Any]], delay_seconds: int = 30) -> None:
    leases = [{"lease_id": msg["lease_id"], "delay_seconds": delay_seconds} for msg in messages if msg.get("lease_id")]
    if leases:
        cf_request("POST", queue_path(account_id, queue_id, "/ack"), token, {"acks": [], "retries": leases})


def decode_message_body(message: dict[str, Any]) -> dict[str, Any]:
    body = message.get("body")
    if isinstance(body, dict):
        return body
    if body is None:
        return {}
    text = str(body)
    metadata = message.get("metadata") or {}
    content_type = metadata.get("CF-Content-Type") or metadata.get("content_type")
    if content_type in {"json", "bytes"}:
        try:
            text = base64.b64decode(text).decode("utf-8")
        except Exception:
            pass
    return json.loads(text)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug[:80] or "task"


def make_task_id(name: str | None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(name or 'cf-colab-task')}"


def parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def build_task(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "id": args.id or make_task_id(args.name),
        "name": args.name,
        "type": "shell",
        "created_at": now(),
        "cmd": args.cmd,
        "cwd": args.cwd,
        "timeout_seconds": args.timeout_seconds,
        "env": parse_env(args.env),
        "artifacts": args.artifact,
        "sync_repo": not args.no_sync_repo,
    }


def run_command(cmd: str, cwd: Path, timeout: int, env: dict[str, str]) -> tuple[int, float, str, str, str | None]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout if timeout > 0 else None,
            env=env,
        )
        return proc.returncode, round(time.monotonic() - started, 3), proc.stdout, proc.stderr, None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, round(time.monotonic() - started, 3), stdout, stderr, f"timeout after {timeout} seconds"


def truncate_text(text: str, limit: int = MAX_EVENT_BODY_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def collect_artifacts(task: dict[str, Any], cwd: Path, max_bytes: int) -> list[dict[str, Any]]:
    artifacts = []
    for pattern in task.get("artifacts", []):
        for item in glob.glob(str(cwd / pattern), recursive=True):
            path = Path(item)
            if not path.is_file() or path.stat().st_size > max_bytes:
                continue
            raw = path.read_bytes()
            artifacts.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size": len(raw),
                    "base64": base64.b64encode(raw).decode("ascii"),
                }
            )
    return artifacts


def send_event(
    account_id: str,
    result_queue_id: str,
    token: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    event = {"task_id": task_id, "event": event_type, "created_at": now(), **payload}
    push_message(account_id, result_queue_id, token, event)


def maybe_sync_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_dir, check=False)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir, check=False)


def execute_task(
    task: dict[str, Any],
    account_id: str,
    result_queue_id: str,
    token: str,
    worker_id: str,
    max_artifact_bytes: int,
) -> None:
    task_id = str(task["id"])
    cwd = Path(task.get("cwd") or "/content/masked-face-id")
    if task.get("sync_repo", True) and cwd == Path("/content/masked-face-id"):
        maybe_sync_repo(cwd)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in task.get("env", {}).items()})
    send_event(
        account_id,
        result_queue_id,
        token,
        task_id,
        "started",
        {"worker_id": worker_id, "cmd": task.get("cmd"), "cwd": str(cwd)},
    )
    returncode, duration, stdout, stderr, error = run_command(
        str(task["cmd"]),
        cwd,
        int(task.get("timeout_seconds") or 0),
        env,
    )
    stdout_tail, stdout_truncated = truncate_text(stdout)
    stderr_tail, stderr_truncated = truncate_text(stderr)
    artifacts = collect_artifacts(task, cwd, max_artifact_bytes)
    send_event(
        account_id,
        result_queue_id,
        token,
        task_id,
        "finished" if returncode == 0 else "failed",
        {
            "worker_id": worker_id,
            "returncode": returncode,
            "duration_seconds": duration,
            "error": error,
            "stdout_tail": stdout_tail,
            "stdout_truncated": stdout_truncated,
            "stderr_tail": stderr_tail,
            "stderr_truncated": stderr_truncated,
            "artifacts": artifacts,
        },
    )


def cmd_submit(args: argparse.Namespace) -> None:
    account_id, task_queue_id, _result_queue_id, token = require_config(args)
    task = build_task(args)
    print(json.dumps(task, indent=2, sort_keys=True))
    if args.dry_run:
        return
    push_message(account_id, task_queue_id, token, task)
    print(f"queued {task['id']} on Cloudflare queue {task_queue_id}")


def cmd_worker(args: argparse.Namespace) -> None:
    account_id, task_queue_id, result_queue_id, token = require_config(args)
    worker_id = args.worker_id or f"colab-{socket.gethostname()}-{os.getpid()}"
    print(json.dumps({"event": "worker_started", "worker_id": worker_id, "task_queue_id": task_queue_id}), flush=True)
    while True:
        messages = pull_messages(
            account_id,
            task_queue_id,
            token,
            batch_size=1,
            visibility_timeout_ms=args.visibility_timeout_ms,
        )
        if not messages:
            time.sleep(args.poll_seconds)
            continue
        message = messages[0]
        try:
            task = decode_message_body(message)
            task.setdefault("id", message.get("id"))
            execute_task(task, account_id, result_queue_id, token, worker_id, args.max_artifact_bytes)
            ack_messages(account_id, task_queue_id, token, [message])
        except Exception as exc:
            task_id = message.get("id", "unknown")
            try:
                send_event(account_id, result_queue_id, token, str(task_id), "worker_error", {"error": repr(exc)})
            finally:
                retry_messages(account_id, task_queue_id, token, [message], delay_seconds=args.retry_delay_seconds)
        if args.once:
            break


def save_result_event(args: argparse.Namespace, event: dict[str, Any]) -> None:
    task_id = str(event.get("task_id") or "unknown")
    task_dir = Path(args.results_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    for artifact in event.get("artifacts", []):
        name = artifact.get("name")
        encoded = artifact.get("base64")
        if name and encoded:
            (task_dir / name).write_bytes(base64.b64decode(encoded))


def cmd_status(args: argparse.Namespace) -> None:
    account_id, _task_queue_id, result_queue_id, token = require_config(args)
    pulled = []
    while True:
        messages = pull_messages(account_id, result_queue_id, token, args.batch_size, args.visibility_timeout_ms)
        if not messages:
            break
        for message in messages:
            event = decode_message_body(message)
            save_result_event(args, event)
            if not args.task_id or event.get("task_id") == args.task_id:
                pulled.append(event)
        ack_messages(account_id, result_queue_id, token, messages)
        if not args.drain:
            break
    if args.json:
        print(json.dumps(pulled, indent=2, sort_keys=True))
        return
    for event in pulled:
        print(
            f"{event.get('created_at')} {event.get('task_id')} {event.get('event')} "
            f"returncode={event.get('returncode')} duration={event.get('duration_seconds')}"
        )
        if args.tail and event.get("stdout_tail"):
            print("--- stdout tail ---")
            print("\n".join(str(event["stdout_tail"]).splitlines()[-args.tail :]))
        if args.tail and event.get("stderr_tail"):
            print("--- stderr tail ---")
            print("\n".join(str(event["stderr_tail"]).splitlines()[-args.tail :]))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--account-id")
    parser.add_argument("--task-queue-id")
    parser.add_argument("--result-queue-id")
    parser.add_argument("--token")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloudflare Queues Colab agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit a shell task to the task queue.")
    add_common(submit)
    submit.add_argument("--cmd", required=True)
    submit.add_argument("--name")
    submit.add_argument("--id")
    submit.add_argument("--cwd", default="/content/masked-face-id")
    submit.add_argument("--timeout-seconds", type=int, default=7200)
    submit.add_argument("--env", action="append", default=[])
    submit.add_argument("--artifact", action="append", default=[])
    submit.add_argument("--no-sync-repo", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=cmd_submit)

    worker = sub.add_parser("worker", help="Run a Colab pull consumer.")
    add_common(worker)
    worker.add_argument("--poll-seconds", type=int, default=15)
    worker.add_argument("--visibility-timeout-ms", type=int, default=43_200_000)
    worker.add_argument("--retry-delay-seconds", type=int, default=60)
    worker.add_argument("--max-artifact-bytes", type=int, default=200_000)
    worker.add_argument("--worker-id")
    worker.add_argument("--once", action="store_true")
    worker.set_defaults(func=cmd_worker)

    status = sub.add_parser("status", help="Drain result events from the result queue.")
    add_common(status)
    status.add_argument("--results-dir", default=".agent_context/cf-colab-results")
    status.add_argument("--task-id")
    status.add_argument("--batch-size", type=int, default=10)
    status.add_argument("--visibility-timeout-ms", type=int, default=60_000)
    status.add_argument("--drain", action="store_true")
    status.add_argument("--json", action="store_true")
    status.add_argument("--tail", type=int, default=40)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
