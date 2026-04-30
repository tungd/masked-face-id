# Synthetic-Train Real-Eval Pair Head Conclusion

Recommendation: MARGINAL

- Train root: `/content/datasets/normalized/lfw_synthetic_mask_pairs`
- Eval root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Baseline full FaceNet masked-unmasked ROC-AUC: 0.7900
- Pair head masked-only masked-unmasked ROC-AUC: 0.7914
- Pair head masked-only gain vs baseline: 0.0014
- Baseline unmasked-unmasked ROC-AUC: 0.9570
- Pair head masked-only unmasked-unmasked ROC-AUC: 0.9570
- Pair head masked-only unmasked regression vs baseline: 0.0000
- Practical masked-unmasked gain threshold: 0.0100

This extension trains the pair-level verifier head on a synthetic LFW
masked/unmasked training root, then evaluates on a separate real-mask RMFRD
identity split. FaceNet remains frozen and unmasked-unmasked pairs are bypassed
to raw FaceNet for the practical masked-only policy.

The result is technically positive but too small to treat as a strong transfer
result. It should be reported as a marginal extension, not as the final project
method.
