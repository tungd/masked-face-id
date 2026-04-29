# RMFRD Occlusion Ensemble Probe

This artifact records a Colab L4 run of `scripts/probe_occlusion_ensemble.py`
on the self-built RMFRD split.

Run parameters:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Train identities: 140
- Eval identities: 80
- Max images per condition: 8
- Pairs per case: 400
- Device: CUDA, Colab L4

Views:

- full face
- lower blackout
- lower blur
- upper-only
- eye-band

Result:

- Full-coverage test-time ensembles did not beat full-face FaceNet on
  masked-unmasked pairs.
- Best full-coverage ensemble: `ensemble_top2_safe`, ROC-AUC `0.7629`
- Full-face baseline masked-unmasked ROC-AUC: `0.7646`
- Reliability/selective verification is more interesting:
  - `gated_upper_only` at 80% coverage: ROC-AUC `0.7807`
  - `ensemble_disagreement_penalty_0.50` at 80% coverage: ROC-AUC `0.7729`

Interpretation: this is not a stronger full-coverage recognizer than the
baseline. It may be useful if the course project is framed around
mask-aware reliability, confidence, and abstention.
