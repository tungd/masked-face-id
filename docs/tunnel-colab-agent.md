# Tunnel Colab Agent

This is the simplest long-job path when MCP is unreliable and the local machine
can stay online. Run a tiny HTTP broker locally, expose it with Cloudflare
Tunnel, then start one Colab worker that polls the public tunnel URL.

Colab only needs the public broker URL and a shared bearer token. It does not
need a GitHub token or a Cloudflare Queues token.

## Security Model

The tunnel URL is public. Always use `TUNNEL_AGENT_TOKEN`; otherwise anyone who
finds the URL can submit shell commands that the Colab worker will run.

Cloudflare Queues themselves are not a good public API surface for this use
case. Keep queues private behind API tokens, or put a small authenticated Worker
or this tunnel broker in front of them.

## Start The Local Broker

From the repo root:

```bash
export TUNNEL_AGENT_TOKEN="$(openssl rand -hex 32)"
python3 scripts/tunnel_colab_agent.py serve --token "$TUNNEL_AGENT_TOKEN"
```

The default broker database is:

```text
.agent_context/tunnel-colab-agent.sqlite3
```

Keep this process running.

## Expose The Broker

In a second local terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Copy the generated `https://...trycloudflare.com` URL:

```bash
export TUNNEL_AGENT_URL="https://example.trycloudflare.com"
```

Quick tunnels change URL every time. Use a named Cloudflare Tunnel and DNS if
you need a stable URL.

## Bootstrap Colab

Add `TUNNEL_AGENT_TOKEN` as a Colab secret, then run one notebook cell:

```python
!rm -rf /content/masked-face-id
!git clone https://github.com/tungd/masked-face-id.git /content/masked-face-id
%cd /content/masked-face-id
!python scripts/tunnel_colab_agent.py worker --broker-url "https://example.trycloudflare.com"
```

If you do not use Colab secrets, pass the token explicitly:

```python
!python scripts/tunnel_colab_agent.py worker \
  --broker-url "https://example.trycloudflare.com" \
  --token "paste-token-here"
```

## Submit Tasks

From the local machine:

```bash
python3 scripts/tunnel_colab_agent.py submit \
  --broker-url "$TUNNEL_AGENT_URL" \
  --token "$TUNNEL_AGENT_TOKEN" \
  --name smoke \
  --cmd 'python scripts/run_validation_spike.py --smoke --results-dir /content/masked_face_spike_results/smoke' \
  --artifact '/content/masked_face_spike_results/smoke/*.csv' \
  --artifact '/content/masked_face_spike_results/smoke/*.md'
```

For a dataset-backed run, mount the dataset in Colab and submit a command that
uses the mounted path.

## Read Results

Status is read directly from the local broker database:

```bash
python3 scripts/tunnel_colab_agent.py status
python3 scripts/tunnel_colab_agent.py status --task-id <task-id> --tail 80
```

Small artifacts are base64-encoded into the final event. Large outputs should be
written to Drive, R2, or another durable store and referenced in stdout.

## Semantics

- The broker stores tasks and result events in local SQLite.
- Workers claim one queued task at a time.
- The worker syncs `/content/masked-face-id` to `origin/main` before each task
  by default.
- If the local broker or tunnel stops, the Colab worker waits until it can poll
  again after restart.
