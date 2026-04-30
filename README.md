# Masked Face Verification

This repository contains a Colab-first research project on masked face
verification. The final direction is a lightweight pair-level verifier that
adapts an existing unmasked FaceNet recognizer without retraining or replacing
the recognizer.

The important verification case is masked-to-unmasked matching: a masked probe
is compared against an existing unmasked gallery image.

## Current Claim

The validated method freezes FaceNet/InceptionResnetV1, extracts several
occlusion-aware embedding views for each image, and trains a small MLP over
pair-level features. The deployed policy is intentionally conservative:

- use the pair head for `masked-masked` and `masked-unmasked` pairs,
- bypass to raw FaceNet cosine similarity for `unmasked-unmasked` pairs.

On held-out RMFD/RMFRD paired identity splits, the main robustness artifact
reports:

| Model | Masked-unmasked ROC-AUC | Unmasked-unmasked ROC-AUC |
|---|---:|---:|
| Raw FaceNet | `0.7972 +/- 0.0027` | `0.9694 +/- 0.0047` |
| Pair head, masked-only policy | `0.8228 +/- 0.0158` | `0.9694 +/- 0.0047` |

The result is a modest but consistent ranking gain. It is not framed as a new
state-of-the-art recognizer. Dedicated mask-aware recognizers remain the
ceiling; this project is about a small adaptation layer for deployments that
already have an unmasked recognizer.

The main limitation is calibration: at a threshold selected on calibration pairs
for nominal FAR `0.05`, the full pair head did not improve TAR over raw FaceNet.
See [docs/final-report.md](docs/final-report.md) for the full result and
[docs/calibration-and-simplification-roadmap.md](docs/calibration-and-simplification-roadmap.md)
for the next technical path.

## Method

For each aligned face, the pipeline extracts FaceNet embeddings from five views:

- full face,
- lower-face blackout,
- lower-face blur,
- upper-only,
- eye-band.

For each candidate pair, the pair head receives:

- one-hot pair type,
- cosine similarities across the five views,
- summary statistics over those scores,
- dense pair interactions: absolute embedding differences and elementwise
  embedding products.

Feature ablations show that dense pair interactions drive the gain. Cosine
scores alone are close to the raw FaceNet baseline.

## Repository Layout

```text
notebooks/
  masked_face_pair_head_final.ipynb     Final pair-head notebook
  smfrd_pair_head_extension.ipynb       Synthetic/SMFRD extension notebook
  validation_spike.ipynb                Earlier first-pass validation spike

scripts/
  probe_pair_head_robustness.py         Main robustness, ablation, threshold run
  probe_pair_verifier_head.py           Single-seed pair-head probe
  probe_maskaware_baseline.py           Dedicated mask-aware ceiling comparison
  probe_insightface_pair_head.py        InsightFace negative control
  analyze_calibration_operating_points.py
                                       Post-run calibration analysis helper
  *_colab_* / *bridge* / *agent*        Local-to-Colab command bridge tools

artifacts/
  Compact checked-in summaries from completed runs

docs/
  final-report.md                       Frozen project report
  reproducibility.md                    Runtime and artifact guidance
  calibration-and-simplification-roadmap.md
                                       Next technical steps
```

## Colab Execution Model

The full GPU experiments are intended to run in Colab. The local machine acts as
the controller, and Colab executes commands through the bridge/tunnel setup:

- local bridge process,
- Cloudflare tunnel exposing that bridge,
- Colab task listening for commands and returning stdout, stderr, and artifacts.

This is documented in:

- [docs/simple-colab-bridge.md](docs/simple-colab-bridge.md)
- [docs/tunnel-colab-agent.md](docs/tunnel-colab-agent.md)
- [docs/cloudflare-colab-agent.md](docs/cloudflare-colab-agent.md)

Use the Colab installer for full notebook/probe runs:

```bash
python scripts/install_colab_deps.py
```

The top-level `requirements.txt` is intentionally Colab-aligned and does not
list packages that Colab already provides, such as `torch`, `numpy`, `pandas`,
or `scikit-learn`.

## Local Checks

For local smoke checks and report utilities, create a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Then run:

```bash
python scripts/validate_notebook.py
python scripts/run_validation_spike.py --smoke --results-dir /tmp/masked-face-id-smoke
```

The smoke run is only a synthetic artifact-writing check. It is not evidence for
or against the masked verification method.

## Main Reproduction Command

After preparing an RMFRD/RMFD-style dataset with both masked and unmasked images
per identity, run the main robustness probe in Colab:

```bash
python scripts/probe_pair_head_robustness.py \
  --data-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_final_runs/rmfd_pair_head_robustness_seed42_7_99 \
  --seeds 42 7 99
```

The primary checked-in artifact is:

- [artifacts/rmfd_pair_head_robustness_seed42_7_99/](artifacts/rmfd_pair_head_robustness_seed42_7_99/)

When raw score CSVs are available from a fresh run, calibration operating points
can be re-analyzed locally:

```bash
python scripts/analyze_calibration_operating_points.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --out-dir /tmp/masked-face-calibration
```

## Next Work

The highest-leverage next steps are:

- refresh calibration so ROC-AUC gains translate into useful fixed-FAR behavior,
- simplify the feature set if full-face dense interactions retain most of the
  gain,
- compare cost and latency against the dedicated mask-aware ceiling,
- add a cleaner run manifest for every Colab experiment.

See [docs/course-demo-finalization-spec.md](docs/course-demo-finalization-spec.md)
for the course-demo plan and [docs/reproducibility.md](docs/reproducibility.md)
for the run checklist.
