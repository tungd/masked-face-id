# Colab Agent Queue

The Colab agent avoids using MCP as the execution channel. MCP only has to run
one small bootstrap cell. After that, a worker inside the Colab runtime polls a
Git-backed queue branch and executes submitted tasks.

## Bootstrap in Colab

Create a Colab secret named `GITHUB_TOKEN` or `COLAB_AGENT_TOKEN` with repo
write access, then run:

```python
!rm -rf /content/masked-face-id
!git clone https://github.com/tungd/masked-face-id.git /content/masked-face-id
%cd /content/masked-face-id
!python scripts/colab_agent_worker.py --push-heartbeats
```

The token is needed because the worker pushes task status, logs, and small
artifacts back to the queue branch.

## Submit a Task Locally

```bash
python3 scripts/colab_agent_submit.py \
  --name feasibility-rmfrd \
  --cmd 'python scripts/run_validation_spike.py --smoke --results-dir /content/masked_face_spike_results/smoke' \
  --artifact '/content/masked_face_spike_results/smoke/*.csv' \
  --artifact '/content/masked_face_spike_results/smoke/*.md'
```

Tasks are written to the `colab-agent-queue` branch under:

```text
.colab-agent/queued/
.colab-agent/running/
.colab-agent/done/
.colab-agent/failed/
.colab-agent/logs/
.colab-agent/results/
```

## Check Status

```bash
python3 scripts/colab_agent_status.py
python3 scripts/colab_agent_status.py --task-id TASK_ID --tail 80
```

## Task Schema

```json
{
  "id": "20260429T090000Z-feasibility-rmfrd",
  "type": "shell",
  "status": "queued",
  "cwd": "/content/masked-face-id",
  "cmd": "python scripts/run_validation_spike.py --smoke",
  "timeout_seconds": 7200,
  "env": {},
  "artifacts": ["results/*.csv", "results/*.md"],
  "sync_repo": true
}
```

`sync_repo` resets `/content/masked-face-id` to `origin/main` before the task
runs. Disable it for tasks that intentionally mutate the working tree.

## Notes

- This is a personal execution queue. It executes shell commands from the queue
  branch, so do not expose the token to untrusted contributors.
- The worker can continue running if MCP disconnects or the browser tab becomes
  hard to automate.
- Heartbeats are pushed only when `--push-heartbeats` is set. Final logs and
  copied artifacts are always pushed when a task finishes.
