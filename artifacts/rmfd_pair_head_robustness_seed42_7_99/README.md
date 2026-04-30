# Pair-Head Robustness, Ablation, and Threshold Probe

This artifact evaluates the FaceNet pair-head method across three identity
splits, feature ablations, and calibration-derived thresholds.

Configuration:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Seeds: `[42, 7, 99]`
- Train identities per seed: 300
- Eval identities per seed: 100
- Max images per condition: 8
- Train pairs per case: 10000
- Eval pairs per case: 2000
- Pair-head epochs: 60
- Feature modes: `full_all_features`, `cosine_scores_only`,
  `cosine_scores_stats`, `dense_interactions_only`, `full_face_dense_only`

Key masked-unmasked ROC-AUC means:

- Raw FaceNet baseline: 0.7972
- Full pair head, masked-only policy: 0.8228
- Full pair head gain: +0.0256

The dense interaction features drive the improvement. At a threshold selected
on calibration pairs for nominal FAR 5%, the full pair-head did not improve TAR
over raw FaceNet; the ranking gain is clearer in ROC-AUC than in the fixed
operating point.
