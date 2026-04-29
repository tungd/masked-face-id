# RMFRD Frozen Adapter Probe

This artifact records a Colab L4 run of `scripts/probe_frozen_adapter.py` on
the self-built RMFRD split.

Run parameters:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Train identities: 140
- Eval identities: 80
- Max images per condition: 8
- Pairs per case: 400
- Ridge: 10.0
- Device: CUDA, Colab L4

Summary:

- Baseline full FaceNet masked-unmasked ROC-AUC: 0.7645
- Best frozen-adapter method: `mean_shift_full_masked_only`
- Best masked-unmasked ROC-AUC: 0.7775
- Gain vs baseline: 0.0130
- Unmasked-unmasked ROC-AUC: unchanged at 0.9710

The result is a weak-positive feasibility signal, not a final claim. The ridge
projection improves masked-masked pairs but hurts masked-unmasked pairs, while
the constrained mean-shift adapter gives the best cross-mask result.
