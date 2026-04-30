# Artifact Index

This directory stores compact summaries from completed masked face verification
experiments. Large raw score logs, datasets, checkpoints, and local backups are
not checked in.

The current main result is the FaceNet pair-head robustness run:

- [rmfd_pair_head_robustness_seed42_7_99/](rmfd_pair_head_robustness_seed42_7_99/)

That artifact evaluates the frozen FaceNet pair-head method across three
identity splits, feature ablations, and calibration-derived thresholds.

| Result | Value |
|---|---:|
| Raw FaceNet masked-unmasked ROC-AUC | `0.7972 +/- 0.0027` |
| Pair head masked-only masked-unmasked ROC-AUC | `0.8228 +/- 0.0158` |
| Mean ROC-AUC gain | `+0.0256` |
| Preserved unmasked-unmasked ROC-AUC | `0.9694 +/- 0.0047` |

The ranking gain is the main positive result. Calibration remains open: at a
threshold selected on calibration pairs for nominal FAR `0.05`, the full
pair-head did not improve TAR over raw FaceNet.

## Main Artifacts

| Path | Purpose |
|---|---|
| `rmfd_pair_head_robustness_seed42_7_99/` | Main three-seed robustness, ablation, and selected threshold summary |
| `insightface_pair_head_seed42/` | InsightFace `buffalo_l` negative control on the seed-42 split |
| `rmfd_pair_verifier_head_full_seed42/` | Single-seed full pair-head run |
| `rmfd_pretrain_adapter_pair_head_seed42/` | Residual adapter pretraining negative result |
| `rmfd_paired_identity_scan/` | Scan showing usable paired RMFD/RMFRD identities |
| `rmfrd_maskaware_baseline_seed42/` | Dedicated mask-aware ceiling, seed 42 |
| `rmfrd_maskaware_baseline_seed7/` | Dedicated mask-aware ceiling, seed 7 |
| `lfw_synthetic_train_rmfrd_eval_seed42/` | Synthetic LFW training with real RMFRD evaluation |

## Earlier Probes

The remaining directories are retained as negative or exploratory probes:

- `rmfrd_frozen_adapter_probe/`
- `rmfrd_occlusion_ensemble_probe/`
- `rmfrd_adaptive_fusion_probe/`
- `rmfrd_arcface_finetune_probe/`
- `rmfrd_periocular_specialist_seed42/`
- `rmfrd_training_adaptation_probe/`
- `rmfrd_pair_verifier_head_probe/`
- `rmfrd_pair_verifier_head_seed7/`
- `rmfrd_cpu_real/`
- `rmfrd_gpu_feasibility/`

These are useful context, but the frozen project claim should be based on
[docs/final-report.md](../docs/final-report.md) and the three-seed robustness
artifact.

## Reproducibility Notes

For future runs, save the raw generated score CSVs outside git or in a release
artifact. They are needed for post-hoc threshold calibration, bootstrap
confidence intervals, and cost/simplification analysis.

Recommended files to retain per run:

- command and commit SHA,
- dataset root and dataset preparation notes,
- train/eval identity manifests,
- sampled pair manifests,
- detector failure counts by condition and split,
- raw pair score CSV,
- trained head checkpoint and standardizer,
- aggregate ROC-AUC and calibration operating-point reports.
