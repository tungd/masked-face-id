# Cloudflare Queue Colab Agent

This is the preferred long-job path when MCP is unreliable and we do not want a
GitHub token in Colab.

MCP only bootstraps one Colab worker. After that, Codex submits tasks to a
Cloudflare Queue, Colab pulls tasks from that queue, and Colab pushes result
events to a second Cloudflare Queue.

## Cloudflare Setup

Create two queues:

```bash
npx wrangler queues create masked-face-colab-tasks
npx wrangler queues create masked-face-colab-results
```

Enable HTTP pull on both queues:

```bash
npx wrangler queues consumer http add masked-face-colab-tasks
npx wrangler queues consumer http add masked-face-colab-results
```

Create a Cloudflare API token with Account -> Queues -> Edit permission. The
same token is used by local submission/status commands and by the Colab worker.

Required variables:

```bash
export CF_ACCOUNT_ID=...
export CF_TASK_QUEUE_ID=...
export CF_RESULT_QUEUE_ID=...
export CF_QUEUES_TOKEN=...
```

Use queue IDs, not queue names, for `CF_TASK_QUEUE_ID` and
`CF_RESULT_QUEUE_ID`.

## Bootstrap Colab

Add the same values as Colab secrets, especially `CF_QUEUES_TOKEN`, then run one
small cell:

```python
!rm -rf /content/masked-face-id
!git clone https://github.com/tungd/masked-face-id.git /content/masked-face-id
%cd /content/masked-face-id
!python scripts/cf_colab_agent.py worker
```

The repo is public, so this does not require a GitHub token.

## Submit Tasks

```bash
python3 scripts/cf_colab_agent.py submit \
  --name smoke \
  --cmd 'python scripts/run_validation_spike.py --smoke --results-dir /content/masked_face_spike_results/smoke' \
  --artifact '/content/masked_face_spike_results/smoke/*.csv' \
  --artifact '/content/masked_face_spike_results/smoke/*.md'
```

## Read Results

```bash
python3 scripts/cf_colab_agent.py status --drain --tail 80
```

Result events are also saved locally under:

```text
.agent_context/cf-colab-results/<task-id>/events.jsonl
```

Small artifacts are base64-encoded into result events and written into the same
task directory. The default per-artifact limit is 200 KB.

## Semantics

- Cloudflare Queues are at-least-once delivery.
- The worker acknowledges a task only after it sends the final result event.
- Set a large visibility timeout for long GPU jobs. The default is 12 hours.
- Very large artifacts should go to Drive/R2/GitHub releases, not through the
  result queue.

Cloudflare docs: https://developers.cloudflare.com/queues/configuration/pull-consumers/
