# Periocular Specialist Probe Conclusion

Recommendation: NOT YET PROMISING

- Tuned fusion alpha: 0.80
- Baseline full FaceNet masked-unmasked ROC-AUC: 0.8219
- Periocular-only masked-unmasked ROC-AUC: 0.6358
- Fixed 0.50 fusion masked-unmasked ROC-AUC: 0.7524
- Tuned fusion masked-unmasked ROC-AUC: 0.6749
- Tuned fusion gain vs baseline: -0.1470
- Baseline unmasked-unmasked ROC-AUC: 0.9819
- Tuned fusion unmasked-unmasked ROC-AUC: 0.9819
- Tuned fusion unmasked regression vs baseline: 0.0000

This probe keeps FaceNet frozen for full-face embeddings, trains a dedicated
identity-supervised embedding head on MediaPipe Face Mesh periocular crops, and
uses the specialist only as fused evidence for masked verification cases.
