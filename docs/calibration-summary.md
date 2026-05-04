# Calibration Summary

The main ranking result is ROC-AUC, but a deployed verifier also needs a
threshold. The checked-in compact artifact includes thresholds selected on
calibration pairs for nominal FAR `0.05` and then applied to held-out eval
pairs.

## FAR 0.05 Operating Point

Source:
`artifacts/rmfd_pair_head_robustness_seed42_7_99/threshold_calibration_selected.csv`

| Model | Eval FAR mean | Eval TAR mean | Eval accuracy mean | Interpretation |
|---|---:|---:|---:|---|
| FaceNet baseline | 0.0481 | 0.3866 | 0.6638 | Best fixed-FAR TAR among these rows |
| Pair head, cosine-only | 0.0495 | 0.3999 | 0.6697 | Small TAR gain, close to baseline behavior |
| Pair head, full features | 0.0502 | 0.3769 | 0.6577 | Better ROC-AUC ranking, worse at this selected threshold |

Seed-level values:

| Seed | Model | FAR | TAR | Accuracy |
|---:|---|---:|---:|---:|
| 7 | FaceNet baseline | 0.0477 | 0.3380 | 0.6395 |
| 42 | FaceNet baseline | 0.0448 | 0.3954 | 0.6657 |
| 99 | FaceNet baseline | 0.0517 | 0.4262 | 0.6862 |
| 7 | Pair head, cosine-only | 0.0456 | 0.3500 | 0.6466 |
| 42 | Pair head, cosine-only | 0.0387 | 0.3764 | 0.6588 |
| 99 | Pair head, cosine-only | 0.0640 | 0.4734 | 0.7037 |
| 7 | Pair head, full features | 0.0560 | 0.3780 | 0.6558 |
| 42 | Pair head, full features | 0.0367 | 0.3346 | 0.6382 |
| 99 | Pair head, full features | 0.0579 | 0.4180 | 0.6790 |

## Interpretation

ROC-AUC and fixed-FAR behavior answer different questions. ROC-AUC measures how
well genuine and impostor pairs are ranked over all possible thresholds. A
fixed-FAR table measures one threshold selected on calibration data and then
applied to held-out evaluation pairs.

The full pair head improves masked-unmasked ranking, but this selected FAR
`0.05` operating point does not improve TAR over raw FaceNet. That is the right
limitation to state in the final presentation: the lightweight pair head is a
useful verifier/ranking adaptation, while deployment calibration still needs
more work.

## Rebuilding Wider FAR Tables

The raw score CSV is intentionally not checked in. When it is available from a
fresh Colab run, regenerate FAR `0.01`, `0.05`, and `0.10` tables with:

```bash
python scripts/analyze_calibration_operating_points.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --out-dir /path/to/calibration_report
```

That command writes:

- `calibration_operating_points.csv`
- `calibration_operating_points_aggregate.csv`
- `calibration_baseline_deltas.csv`
- `calibration_summary.json`
