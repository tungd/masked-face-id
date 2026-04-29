# Simple Colab Bridge

This is the dumb path for using a Colab runtime from Codex:

1. Run a local in-memory bridge.
2. Expose it with Cloudflare Tunnel.
3. Run one receiver cell in Colab.
4. Submit one command locally and wait for stdout, stderr, and return code.

There is no durable queue, database, retry policy, or artifact store. If the
local bridge dies, the in-flight command state is gone.

## Start The Local Bridge

For a short-lived random Cloudflare Tunnel URL, the low-friction path is to run
without a token and tear the tunnel down when finished:

```bash
python3 scripts/colab_bridge.py serve --allow-unauthenticated
```

Keep this process running.

If you want auth on the public URL, use a shared token instead:

```bash
export COLAB_BRIDGE_TOKEN="$(openssl rand -hex 32)"
python3 scripts/colab_bridge.py serve --token "$COLAB_BRIDGE_TOKEN"
```

## Expose It

In another local terminal:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

Copy the generated URL:

```bash
export COLAB_BRIDGE_URL="https://example.trycloudflare.com"
```

## Start The Colab Receiver

For the no-token path, run:

```python
!rm -rf /content/masked-face-id
!git clone https://github.com/tungd/masked-face-id.git /content/masked-face-id
%cd /content/masked-face-id
!python scripts/colab_bridge.py receive --url "https://example.trycloudflare.com"
```

For the token path, add `COLAB_BRIDGE_TOKEN` as a Colab secret and use the same
receiver command. If you do not use Colab secrets, pass the token explicitly:

```python
!python scripts/colab_bridge.py receive \
  --url "https://example.trycloudflare.com" \
  --token "paste-token-here"
```

## Run A Command

From the local machine:

```bash
python3 scripts/colab_bridge.py run \
  --url "$COLAB_BRIDGE_URL" \
  --cmd 'python scripts/run_validation_spike.py --smoke'
```

Add `--token "$COLAB_BRIDGE_TOKEN"` when running the token path.

For a long command, increase both the command timeout and local wait timeout:

```bash
python3 scripts/colab_bridge.py run \
  --url "$COLAB_BRIDGE_URL" \
  --timeout-seconds 21600 \
  --wait-seconds 21600 \
  --cmd 'python scripts/run_validation_spike.py --dataset /content/pku_masked_face_subset'
```

## Notes

- The bridge accepts one command at a time.
- The response is limited to stdout and stderr tails, capped in the script.
- Write large outputs to Drive or another mounted location.
- The no-token path relies on the random tunnel URL being temporary. Tear down
  `cloudflared` and the local bridge when finished.
