# Reproducibility

This project is Colab-first because the full FaceNet embedding and pair-head
runs need GPU access. The local checkout is the controller and artifact
workspace; Colab is the execution runtime.

## Runtime Split

Use this split intentionally:

- local machine: edit code, run static checks, run smoke/report utilities,
  receive artifacts,
- Colab: install runtime extras, mount/download datasets, run GPU probes,
  write raw scores and model artifacts,
- bridge/tunnel: submit commands from local to Colab when Colab MCP is
  unreliable.

The bridge workflows are documented in:

- [simple-colab-bridge.md](simple-colab-bridge.md)
- [tunnel-colab-agent.md](tunnel-colab-agent.md)
- [cloudflare-colab-agent.md](cloudflare-colab-agent.md)

## Dependencies

For Colab:

```bash
python scripts/install_colab_deps.py
```

For local smoke checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Do not use the local requirements file as the Colab benchmark setup. Colab has
preinstalled `torch`, `numpy`, and related packages; resolving those through pip
can change the runtime stack.

## Minimal Checks

Local:

```bash
python scripts/validate_notebook.py
python scripts/run_validation_spike.py --smoke --results-dir /tmp/masked-face-id-smoke
```

Colab:

```bash
python scripts/probe_pair_head_robustness.py \
  --data-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_final_runs/rmfd_pair_head_robustness_seed42_7_99 \
  --seeds 42 7 99
```

Post-run calibration analysis, when raw scores are available:

```bash
python scripts/analyze_calibration_operating_points.py \
  --scores /content/masked_face_final_runs/rmfd_pair_head_robustness_seed42_7_99/pair_head_robustness_pair_scores.csv \
  --out-dir /content/masked_face_final_runs/rmfd_pair_head_robustness_seed42_7_99/calibration
```

Offline demo export, when raw scores and the dataset image root are available:

```bash
python scripts/export_demo_bundle.py \
  --scores /content/masked_face_final_runs/rmfd_pair_head_robustness_seed42_7_99/pair_head_robustness_pair_scores.csv \
  --image-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_final_runs/demo_bundle \
  --seed 42 \
  --target-far 0.05
```

Download or copy the resulting `demo_bundle/` directory and open
`demo_bundle/index.html` for the presentation-safe curated verifier. The app
uses precomputed scores and copied images, so it does not need Colab, a GPU, or
network access during the demo.

## Run Manifest Checklist

Save these with every full run:

- git commit SHA,
- exact command,
- runtime type and GPU name,
- dataset source and normalized dataset root,
- train/eval identity manifests,
- sampled pair manifests,
- image counts by split and condition,
- detector failures by split and condition,
- raw pair scores for calibration and bootstrap analysis,
- trained pair-head checkpoint,
- feature standardizer,
- aggregate ROC-AUC report,
- operating-point report for FAR `0.01`, `0.05`, and `0.10`.
- exported offline demo bundle, or enough raw scores and images to rebuild it.

The checked-in `artifacts/` directory should stay compact. Large raw score logs,
checkpoints, downloaded datasets, and local backups should remain outside git.
