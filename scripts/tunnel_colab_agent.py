#!/usr/bin/env python3
"""Local HTTP broker + Colab worker for tunnel-based task execution.

Run the broker locally, expose it with cloudflared, then point a Colab worker at
the public tunnel URL. This keeps credentials out of Colab except for a shared
broker token.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import json
import os
import re
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


MAX_EVENT_TEXT_CHARS = 95_000


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug[:80] or "task"


def make_task_id(name: str | None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(name or 'tunnel-task')}"


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
        "status": "queued",
        "cmd": args.cmd,
        "cwd": args.cwd,
        "timeout_seconds": args.timeout_seconds,
        "env": parse_env(args.env),
        "artifacts": args.artifact,
        "sync_repo": not args.no_sync_repo,
    }


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute(
        """
        create table if not exists tasks (
            id text primary key,
            status text not null,
            payload text not null,
            created_at text not null,
            updated_at text not null,
            worker_id text,
            returncode integer,
            duration_seconds real
        )
        """
    )
    db.execute(
        """
        create table if not exists events (
            id integer primary key autoincrement,
            task_id text not null,
            event text not null,
            payload text not null,
            created_at text not null
        )
        """
    )
    db.commit()
    return db


def task_from_row(row: sqlite3.Row) -> dict[str, Any]:
    task = json.loads(row["payload"])
    task["status"] = row["status"]
    task["updated_at"] = row["updated_at"]
    if row["worker_id"]:
        task["worker_id"] = row["worker_id"]
    if row["returncode"] is not None:
        task["returncode"] = row["returncode"]
    if row["duration_seconds"] is not None:
        task["duration_seconds"] = row["duration_seconds"]
    return task


def insert_task(db: sqlite3.Connection, task: dict[str, Any]) -> None:
    db.execute(
        "insert into tasks(id, status, payload, created_at, updated_at) values (?, ?, ?, ?, ?)",
        (task["id"], "queued", json.dumps(task, sort_keys=True), task["created_at"], now()),
    )
    db.commit()


def get_tasks(db: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = db.execute("select * from tasks order by created_at").fetchall()
    return [task_from_row(row) for row in rows]


def get_task(db: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    row = db.execute("select * from tasks where id = ?", (task_id,)).fetchone()
    return task_from_row(row) if row else None


def task_events(db: sqlite3.Connection, task_id: str | None = None) -> list[dict[str, Any]]:
    if task_id:
        rows = db.execute("select * from events where task_id = ? order by id", (task_id,)).fetchall()
    else:
        rows = db.execute("select * from events order by id").fetchall()
    events = []
    for row in rows:
        payload = json.loads(row["payload"])
        payload.setdefault("task_id", row["task_id"])
        payload.setdefault("event", row["event"])
        payload.setdefault("created_at", row["created_at"])
        events.append(payload)
    return events


def claim_next_task(db: sqlite3.Connection, worker_id: str) -> dict[str, Any] | None:
    row = db.execute("select * from tasks where status = 'queued' order by created_at limit 1").fetchone()
    if not row:
        return None
    task = task_from_row(row)
    task["status"] = "running"
    task["worker_id"] = worker_id
    task["started_at"] = now()
    db.execute(
        "update tasks set status = 'running', payload = ?, updated_at = ?, worker_id = ? where id = ? and status = 'queued'",
        (json.dumps(task, sort_keys=True), now(), worker_id, task["id"]),
    )
    db.commit()
    changed = db.execute("select changes()").fetchone()[0]
    return task if changed else None


def add_event(db: sqlite3.Connection, event: dict[str, Any]) -> None:
    task_id = str(event.get("task_id"))
    event_type = str(event.get("event"))
    created_at = event.get("created_at") or now()
    db.execute(
        "insert into events(task_id, event, payload, created_at) values (?, ?, ?, ?)",
        (task_id, event_type, json.dumps(event, sort_keys=True), created_at),
    )
    if event_type in {"finished", "failed", "worker_error"}:
        status = "done" if event_type == "finished" and event.get("returncode") == 0 else "failed"
        row = db.execute("select * from tasks where id = ?", (task_id,)).fetchone()
        if row:
            task = task_from_row(row)
            task["status"] = status
            task["finished_at"] = created_at
            task["returncode"] = event.get("returncode")
            task["duration_seconds"] = event.get("duration_seconds")
            db.execute(
                "update tasks set status = ?, payload = ?, updated_at = ?, returncode = ?, duration_seconds = ? where id = ?",
                (
                    status,
                    json.dumps(task, sort_keys=True),
                    now(),
                    event.get("returncode"),
                    event.get("duration_seconds"),
                    task_id,
                ),
            )
    db.commit()


class BrokerHandler(BaseHTTPRequestHandler):
    server_version = "TunnelColabAgent/0.1"

    @property
    def broker(self) -> "BrokerServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: object) -> None:
        if self.broker.verbose:
            super().log_message(format, *args)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        expected = self.broker.token
        if not expected:
            return True
        auth = self.headers.get("authorization", "")
        return auth == f"Bearer {expected}"

    def require_auth(self) -> bool:
        if self.authorized():
            return True
        self.send_json(401, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"ok": True, "time": now()})
            return
        if not self.require_auth():
            return
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/tasks":
            self.send_json(200, {"tasks": get_tasks(self.broker.db)})
            return
        if parsed.path == "/events":
            task_id = params.get("task_id", [None])[0]
            self.send_json(200, {"events": task_events(self.broker.db, task_id)})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.require_auth():
            return
        if self.path == "/tasks":
            task = self.read_json()
            task.setdefault("id", make_task_id(task.get("name")))
            task.setdefault("created_at", now())
            task.setdefault("status", "queued")
            insert_task(self.broker.db, task)
            self.send_json(200, {"task": task})
            return
        if self.path == "/poll":
            payload = self.read_json()
            task = claim_next_task(self.broker.db, str(payload.get("worker_id") or "unknown"))
            self.send_json(200, {"task": task})
            return
        if self.path == "/events":
            event = self.read_json()
            event.setdefault("created_at", now())
            add_event(self.broker.db, event)
            self.send_json(200, {"ok": True})
            return
        self.send_json(404, {"error": "not found"})


class BrokerServer(HTTPServer):
    def __init__(self, address: tuple[str, int], db: sqlite3.Connection, token: str | None, verbose: bool):
        super().__init__(address, BrokerHandler)
        self.db = db
        self.token = token
        self.verbose = verbose


def cmd_serve(args: argparse.Namespace) -> None:
    db = connect_db(Path(args.db))
    token = args.token or os.environ.get("TUNNEL_AGENT_TOKEN")
    if not token and not args.allow_unauthenticated:
        raise SystemExit("Refusing to start without --token or TUNNEL_AGENT_TOKEN. Use --allow-unauthenticated for local-only testing.")
    server = BrokerServer((args.host, args.port), db, token, args.verbose)
    print(json.dumps({"event": "broker_started", "host": args.host, "port": args.port, "db": args.db}), flush=True)
    server.serve_forever()


def broker_request(method: str, broker_url: str, token: str | None, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(broker_url.rstrip("/") + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"broker {method} {path} failed: {exc.code} {detail}") from exc


def cmd_submit(args: argparse.Namespace) -> None:
    task = build_task(args)
    print(json.dumps(task, indent=2, sort_keys=True))
    if args.dry_run:
        return
    response = broker_request("POST", args.broker_url, args.token or os.environ.get("TUNNEL_AGENT_TOKEN"), "/tasks", task)
    print(f"queued {response['task']['id']} at {args.broker_url}")


def truncate_text(text: str, limit: int = MAX_EVENT_TEXT_CHARS) -> tuple[str, bool]:
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
            artifacts.append({"name": path.name, "path": str(path), "size": len(raw), "base64": base64.b64encode(raw).decode("ascii")})
    return artifacts


def post_event(broker_url: str, token: str | None, event: dict[str, Any]) -> None:
    broker_request("POST", broker_url, token, "/events", event)


def maybe_sync_repo(repo_dir: Path) -> None:
    subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_dir, check=False)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=repo_dir, check=False)


def run_task(task: dict[str, Any], args: argparse.Namespace, worker_id: str) -> None:
    task_id = str(task["id"])
    cwd = Path(task.get("cwd") or "/content/masked-face-id")
    if task.get("sync_repo", True) and cwd == Path("/content/masked-face-id"):
        maybe_sync_repo(cwd)
    token = args.token or env_or_secret("TUNNEL_AGENT_TOKEN")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in task.get("env", {}).items()})
    post_event(args.broker_url, token, {"task_id": task_id, "event": "started", "created_at": now(), "worker_id": worker_id, "cmd": task.get("cmd"), "cwd": str(cwd)})
    started = time.monotonic()
    try:
        proc = subprocess.run(
            str(task["cmd"]),
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=int(task.get("timeout_seconds") or 0) or None,
            env=env,
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        error = f"timeout after {task.get('timeout_seconds')} seconds"
    stdout_tail, stdout_truncated = truncate_text(stdout)
    stderr_tail, stderr_truncated = truncate_text(stderr)
    post_event(
        args.broker_url,
        token,
        {
            "task_id": task_id,
            "event": "finished" if returncode == 0 else "failed",
            "created_at": now(),
            "worker_id": worker_id,
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": error,
            "stdout_tail": stdout_tail,
            "stdout_truncated": stdout_truncated,
            "stderr_tail": stderr_tail,
            "stderr_truncated": stderr_truncated,
            "artifacts": collect_artifacts(task, cwd, args.max_artifact_bytes),
        },
    )


def cmd_worker(args: argparse.Namespace) -> None:
    token = args.token or env_or_secret("TUNNEL_AGENT_TOKEN")
    worker_id = args.worker_id or f"colab-{socket.gethostname()}-{os.getpid()}"
    print(json.dumps({"event": "worker_started", "worker_id": worker_id, "broker_url": args.broker_url}), flush=True)
    while True:
        try:
            response = broker_request("POST", args.broker_url, token, "/poll", {"worker_id": worker_id})
        except Exception as exc:
            print(json.dumps({"event": "poll_error", "error": repr(exc), "time": now()}), flush=True)
            time.sleep(args.poll_seconds)
            continue
        task = response.get("task")
        if not task:
            time.sleep(args.poll_seconds)
            continue
        run_task(task, args, worker_id)
        if args.once:
            break


def cmd_status(args: argparse.Namespace) -> None:
    db = connect_db(Path(args.db))
    if args.task_id:
        task = get_task(db, args.task_id)
        events = task_events(db, args.task_id)
        print(json.dumps({"task": task, "events": events}, indent=2, sort_keys=True))
        if args.tail:
            for event in events[-args.tail :]:
                print(f"{event.get('created_at')} {event.get('event')} returncode={event.get('returncode')}")
                if event.get("stdout_tail"):
                    print("--- stdout tail ---")
                    print(event["stdout_tail"])
                if event.get("stderr_tail"):
                    print("--- stderr tail ---")
                    print(event["stderr_tail"])
        return
    print(json.dumps({"tasks": get_tasks(db)}, indent=2, sort_keys=True))


def add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--name")
    parser.add_argument("--id")
    parser.add_argument("--cwd", default="/content/masked-face-id")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--no-sync-repo", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tunnel-backed Colab task agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local task broker.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--db", default=".agent_context/tunnel-colab-agent.sqlite3")
    serve.add_argument("--token", default=None)
    serve.add_argument("--allow-unauthenticated", action="store_true")
    serve.add_argument("--verbose", action="store_true")
    serve.set_defaults(func=cmd_serve)

    submit = sub.add_parser("submit", help="Submit a task to the broker.")
    submit.add_argument("--broker-url", required=True)
    submit.add_argument("--token", default=None)
    submit.add_argument("--dry-run", action="store_true")
    add_task_args(submit)
    submit.set_defaults(func=cmd_submit)

    worker = sub.add_parser("worker", help="Run the Colab polling worker.")
    worker.add_argument("--broker-url", required=True)
    worker.add_argument("--token", default=None)
    worker.add_argument("--poll-seconds", type=int, default=10)
    worker.add_argument("--max-artifact-bytes", type=int, default=200_000)
    worker.add_argument("--worker-id")
    worker.add_argument("--once", action="store_true")
    worker.set_defaults(func=cmd_worker)

    status = sub.add_parser("status", help="Inspect local broker state.")
    status.add_argument("--db", default=".agent_context/tunnel-colab-agent.sqlite3")
    status.add_argument("--task-id")
    status.add_argument("--tail", type=int, default=0)
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
