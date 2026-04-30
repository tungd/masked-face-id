# Calibration And Simplification Roadmap

The pair head has a repeatable ROC-AUC gain, but the deployment problem is an
operating point problem: at a fixed allowed false accept rate, we need higher
true accept rate without harming unmasked-unmasked behavior.

## Known Baseline

From the three-seed robustness artifact:

- raw FaceNet masked-unmasked ROC-AUC: `0.7972 +/- 0.0027`,
- full pair head masked-unmasked ROC-AUC: `0.8228 +/- 0.0158`,
- unmasked-unmasked ROC-AUC is preserved by bypassing the head,
- at calibration-derived FAR `0.05`, the full pair head did not improve TAR.

That means the next work should prioritize calibration and feature economy, not
another broad architecture search.

## Calibration Work

Run calibration analysis from raw score CSVs with:

```bash
python scripts/analyze_calibration_operating_points.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --out-dir /path/to/calibration_report
```

Evaluate each model and pair case at:

- FAR `0.01`,
- FAR `0.05`,
- FAR `0.10`,
- best calibration accuracy.

Then add these variants:

- case-specific thresholds for `masked-masked` and `masked-unmasked`,
- Platt scaling trained only on calibration pairs,
- isotonic regression trained only on calibration pairs,
- bootstrap confidence intervals over eval pairs and over identity splits,
- DET/EER reporting alongside ROC-AUC.

The decision criterion should be practical: keep a calibration method only if it
improves masked-unmasked TAR at the target FAR while preserving the existing
unmasked-unmasked bypass.

## Feature Simplification

The current ablation shows that full-face dense interactions are close to the
full five-view feature set:

| Feature set | Masked-unmasked ROC-AUC |
|---|---:|
| Full five-view features + dense interactions | `0.8228 +/- 0.0158` |
| Full-face dense interactions only | `0.8185 +/- 0.0205` |
| Cosine scores only | `0.7986 +/- 0.0021` |
| Raw FaceNet baseline | `0.7972 +/- 0.0027` |

Next simplification runs:

- full-face dense interactions only,
- full face plus lower-blur dense interactions,
- full face plus upper-only dense interactions,
- no score statistics,
- smaller hidden dimension,
- fewer epochs with early stopping.

Report:

- ROC-AUC by case,
- TAR at FAR `0.01`, `0.05`, and `0.10`,
- embedding views per image,
- feature dimension,
- pair-head parameter count,
- approximate inference latency.

If the one-view or two-view head keeps most of the gain, prefer it. A smaller
adapter is easier to justify as an operational layer on top of a frozen
recognizer.
