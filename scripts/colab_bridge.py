#!/usr/bin/env python3
"""Minimal command bridge between local Codex and a Colab runtime.

Run `serve` locally, expose it with cloudflared, then run `receive` in Colab.
Local `run` calls submit one command, wait for Colab to execute it, and print
the result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition
from typing import Any


MAX_TEXT_CHARS = 200_000


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug[:80] or "cmd"


def make_command_id(name: str | None) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slugify(name or 'colab-command')}"


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
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def truncate_text(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


class BridgeState:
    def __init__(self) -> None:
        self.condition = Condition()
        self.pending: dict[str, Any] | None = None
        self.running: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None

    def submit(self, command: dict[str, Any], wait_seconds: int) -> tuple[int, dict[str, Any]]:
        with self.condition:
            if self.pending or self.running:
                current = self.running or self.pending
                return 409, {"error": "busy", "current_id": current.get("id") if current else None}
            self.pending = command
            self.condition.notify_all()
            deadline = time.monotonic() + wait_seconds
            while True:
                if self.last_result and self.last_result.get("id") == command["id"]:
                    result = self.last_result
                    return 200, {"result": result}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 202, {"status": "running_or_waiting", "id": command["id"]}
                self.condition.wait(timeout=min(remaining, 5))

    def next_command(self, wait_seconds: int) -> dict[str, Any] | None:
        with self.condition:
            deadline = time.monotonic() + wait_seconds
            while not self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(timeout=min(remaining, 5))
            command = self.pending
            self.pending = None
            self.running = command
            self.condition.notify_all()
            return command

    def finish(self, result: dict[str, Any]) -> None:
        with self.condition:
            self.running = None
            self.last_result = result
            self.condition.notify_all()


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "ColabBridge/0.1"

    @property
    def bridge(self) -> "BridgeServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: object) -> None:
        if self.bridge.verbose:
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
        expected = self.bridge.token
        if not expected:
            return True
        return self.headers.get("authorization", "") == f"Bearer {expected}"

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
        if self.path == "/last":
            self.send_json(200, {"result": self.bridge.state.last_result})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self.require_auth():
            return
        if self.path == "/run":
            payload = self.read_json()
            command = {
                "id": payload.get("id") or make_command_id(payload.get("name")),
                "name": payload.get("name"),
                "cmd": payload["cmd"],
                "cwd": payload.get("cwd") or "/content/masked-face-id",
                "env": payload.get("env") or {},
                "timeout_seconds": int(payload.get("timeout_seconds") or 7200),
                "created_at": now(),
            }
            status, response = self.bridge.state.submit(command, int(payload.get("wait_seconds") or 7200))
            self.send_json(status, response)
            return
        if self.path == "/next":
            payload = self.read_json()
            command = self.bridge.state.next_command(int(payload.get("wait_seconds") or 30))
            self.send_json(200, {"command": command})
            return
        if self.path == "/result":
            result = self.read_json()
            result.setdefault("finished_at", now())
            self.bridge.state.finish(result)
            self.send_json(200, {"ok": True})
            return
        self.send_json(404, {"error": "not found"})


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], token: str | None, verbose: bool):
        super().__init__(address, BridgeHandler)
        self.token = token
        self.verbose = verbose
        self.state = BridgeState()


def request_json(method: str, base_url: str, token: str | None, path: str, body: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"bridge {method} {path} failed: {exc.code} {detail}") from exc


def cmd_serve(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("COLAB_BRIDGE_TOKEN")
    if not token and not args.allow_unauthenticated:
        raise SystemExit("Refusing to start without --token or COLAB_BRIDGE_TOKEN. Use --allow-unauthenticated for local-only testing.")
    server = BridgeServer((args.host, args.port), token, args.verbose)
    print(json.dumps({"event": "bridge_started", "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()


def cmd_run(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("COLAB_BRIDGE_TOKEN")
    payload = {
        "id": args.id or make_command_id(args.name),
        "name": args.name,
        "cmd": args.cmd,
        "cwd": args.cwd,
        "env": parse_env(args.env),
        "timeout_seconds": args.timeout_seconds,
        "wait_seconds": args.wait_seconds,
    }
    response = request_json("POST", args.url, token, "/run", payload, timeout=args.wait_seconds + 30)
    result = response.get("result")
    if not result:
        print(json.dumps(response, indent=2, sort_keys=True))
        return
    if result.get("stdout"):
        print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")
    if result.get("stderr"):
        print(result["stderr"], end="" if result["stderr"].endswith("\n") else "\n", file=sys.stderr)
    if result.get("error"):
        print(result["error"], file=sys.stderr)
    raise SystemExit(int(result.get("returncode") or 0))


def execute_command(command: dict[str, Any]) -> dict[str, Any]:
    cwd = Path(command.get("cwd") or "/content/masked-face-id")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in command.get("env", {}).items()})
    started = time.monotonic()
    try:
        proc = subprocess.run(
            str(command["cmd"]),
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=int(command.get("timeout_seconds") or 0) or None,
            env=env,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        error = None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = 124
        error = f"timeout after {command.get('timeout_seconds')} seconds"
    stdout, stdout_truncated = truncate_text(stdout)
    stderr, stderr_truncated = truncate_text(stderr)
    return {
        "id": command["id"],
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "stderr": stderr,
        "stderr_truncated": stderr_truncated,
        "error": error,
        "worker_id": f"colab-{socket.gethostname()}-{os.getpid()}",
    }


def cmd_receive(args: argparse.Namespace) -> None:
    token = args.token or env_or_secret("COLAB_BRIDGE_TOKEN")
    print(json.dumps({"event": "receiver_started", "url": args.url}), flush=True)
    while True:
        try:
            response = request_json("POST", args.url, token, "/next", {"wait_seconds": args.poll_wait_seconds}, timeout=args.poll_wait_seconds + 10)
        except Exception as exc:
            print(json.dumps({"event": "poll_error", "error": repr(exc), "time": now()}), flush=True)
            time.sleep(args.retry_seconds)
            continue
        command = response.get("command")
        if not command:
            continue
        print(json.dumps({"event": "command_started", "id": command.get("id"), "cmd": command.get("cmd")}), flush=True)
        result = execute_command(command)
        request_json("POST", args.url, token, "/result", result, timeout=60)
        print(json.dumps({"event": "command_finished", "id": command.get("id"), "returncode": result["returncode"]}), flush=True)
        if args.once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Colab command bridge.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local in-memory bridge.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8876)
    serve.add_argument("--token")
    serve.add_argument("--allow-unauthenticated", action="store_true")
    serve.add_argument("--verbose", action="store_true")
    serve.set_defaults(func=cmd_serve)

    run = sub.add_parser("run", help="Submit one command and wait for the result.")
    run.add_argument("--url", required=True)
    run.add_argument("--token")
    run.add_argument("--cmd", required=True)
    run.add_argument("--name")
    run.add_argument("--id")
    run.add_argument("--cwd", default="/content/masked-face-id")
    run.add_argument("--timeout-seconds", type=int, default=7200)
    run.add_argument("--wait-seconds", type=int, default=7200)
    run.add_argument("--env", action="append", default=[])
    run.set_defaults(func=cmd_run)

    receive = sub.add_parser("receive", help="Run the Colab receive loop.")
    receive.add_argument("--url", required=True)
    receive.add_argument("--token")
    receive.add_argument("--poll-wait-seconds", type=int, default=30)
    receive.add_argument("--retry-seconds", type=int, default=5)
    receive.add_argument("--once", action="store_true")
    receive.set_defaults(func=cmd_receive)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
