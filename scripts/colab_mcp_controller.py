#!/usr/bin/env python3
"""Interactive Colab MCP controller for manual recovery/debug sessions.

It starts a fresh Colab MCP websocket bridge, opens the scratch notebook URL,
prints the dynamically exposed Colab tools, then accepts JSON lines on stdin:

{"tool": "tool_name", "args": {"name": "value"}}
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import colab_mcp.session as colab_session
from colab_mcp.session import ColabSessionProxy
from colab_mcp.websocket_server import COLAB, SCRATCH_PATH


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump_json"):
        try:
            return json.loads(value.model_dump_json(by_alias=True))
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", by_alias=True)
        except Exception:
            pass
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {str(key): to_jsonable(item) for key, item in vars(value).items()}
    return repr(value)


async def main() -> None:
    colab_session.UI_CONNECTION_TIMEOUT = 300.0
    call_timeout = float(os.environ.get("COLAB_MCP_CALL_TIMEOUT", "60"))

    session = ColabSessionProxy()
    await session.start_proxy_server()
    proxy_client = session.middleware[0].proxy_client
    url = (
        f"{COLAB}{SCRATCH_PATH}"
        f"#mcpProxyToken={session.wss.token}&mcpProxyPort={session.wss.port}"
    )

    print(json.dumps({"event": "open_url", "url": url}), flush=True)
    subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    print(json.dumps({"event": "waiting_for_colab", "timeout_seconds": 300}), flush=True)

    await proxy_client.await_proxy_connection()
    connected = proxy_client.is_connected()
    print(json.dumps({"event": "connected", "connected": connected}), flush=True)
    if not connected:
        await session.cleanup()
        return

    tools = await proxy_client.proxy_mcp_client.list_tools()
    print(
        json.dumps(
            {
                "event": "tools",
                "call_timeout_seconds": call_timeout,
                "tools": [
                    {
                        "name": tool.name,
                        "description": getattr(tool, "description", None),
                        "input_schema": getattr(tool, "input_schema", None),
                    }
                    for tool in tools
                ],
            }
        ),
        flush=True,
    )

    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            break
        try:
            command = json.loads(line)
            if "args_file" in command:
                args_path = Path(command["args_file"]).expanduser()
                command["args"] = json.loads(args_path.read_text())
            started = time.monotonic()
            result = await asyncio.wait_for(
                proxy_client.proxy_mcp_client.call_tool(
                    command["tool"], command.get("args", {})
                ),
                timeout=float(command.get("timeout", call_timeout)),
            )
            print(
                json.dumps(
                    {
                        "event": "tool_result",
                        "tool": command["tool"],
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "result": to_jsonable(result),
                    },
                    default=repr,
                ),
                flush=True,
            )
        except asyncio.TimeoutError:
            print(
                json.dumps(
                    {
                        "event": "timeout",
                        "tool": command.get("tool"),
                        "timeout_seconds": command.get("timeout", call_timeout),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps({"event": "error", "error": repr(exc)}),
                flush=True,
            )

    await session.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
