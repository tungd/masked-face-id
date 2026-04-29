# RMFRD Adaptive Fusion Probe

This probe reuses the L4 GPU run's saved pair scores from
`/content/masked_face_spike_results/feasibility_gpu/validation_pair_scores.csv`.
It does not recompute embeddings.

Goal: test whether score-level policies around an unmasked FaceNet recognizer
show enough signal to justify the project framing.

## Result

There is a **promising but weak signal**.

- Baseline masked-unmasked ROC-AUC: `0.9275`
- Best policy: `mask_presence_gated_blackout`
- Best policy masked-unmasked ROC-AUC: `0.9408`
- Gain: `+0.0133`
- Baseline unmasked-unmasked ROC-AUC: `0.9544`
- Best policy unmasked-unmasked ROC-AUC: `0.9544`
- Unmasked regression: `0.0000`

The useful policy is simple: use lower-face blackout when a pair involves a
masked image, and keep the full-face baseline for unmasked-unmasked pairs. This
preserves unmasked performance because it does not apply occlusion preprocessing
where there is no mask.

## Caveat

The learned logistic-regression fusion policies did **not** improve
masked-unmasked ROC-AUC. The calibration-selected `case_gated_best_raw` policy
also selected the baseline for masked-unmasked, so the positive result currently
depends on a manually specified mask-presence gate.

This is enough to continue probing the idea, but not yet enough for the final
claim. The next required comparison is against a real mask-aware recognizer.
