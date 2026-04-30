# Pair Verifier Head Probe Conclusion

Recommendation: PROMISING

- Baseline full FaceNet masked-unmasked ROC-AUC: 0.7999
- Best candidate: pair_head_masked_cases_only
- Best masked-unmasked ROC-AUC: 0.8214
- Best gain vs baseline: 0.0215
- Baseline unmasked-unmasked ROC-AUC: 0.9687
- Best unmasked-unmasked ROC-AUC: 0.9687
- Best unmasked regression vs baseline: 0.0000

This probe freezes FaceNet and trains only a pair-level MLP verifier head over
multi-view embedding features.
