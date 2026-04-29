# Frozen Adapter Probe Conclusion

Recommendation: PROMISING

- Best method on masked-unmasked: mean_shift_full_masked_only
- Masked-unmasked ROC-AUC baseline: 0.7645
- Masked-unmasked ROC-AUC best method: 0.7775
- Masked-unmasked gain vs baseline: 0.0130
- Unmasked-unmasked ROC-AUC baseline: 0.9710
- Unmasked-unmasked ROC-AUC best method: 0.9710
- Unmasked regression vs baseline: 0.0000

This probe trains lightweight adapters on calibration identities only and
evaluates on held-out identities. FaceNet remains frozen. Tested adapters are
ridge-linear projection, orthogonal projection, and mean shift.
