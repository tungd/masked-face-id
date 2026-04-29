# Validation Spike Conclusion

Recommendation: NO-GO

- Dataset: RMFRD fallback from Real-World-Masked-Face-Dataset.
- Run: Colab CPU, 100 identities, 300 requested pairs per case.
- Effective evaluated pairs were lower because detector failures were skipped.
- Masked-unmasked ROC-AUC baseline: 0.8683
- Masked-unmasked ROC-AUC upper-face: 0.7583
- Masked-unmasked gain: -0.1100
- Unmasked-unmasked ROC-AUC baseline: 0.9598
- Unmasked-unmasked ROC-AUC upper-face: 0.8318
- Unmasked regression: 0.1279

Decision rule: go only if masked-unmasked ROC-AUC improves by at least 0.03 and unmasked-unmasked ROC-AUC drops by no more than 0.02.

Third model note: default candidate is FaceNet CASIA-WebFace; replace with MaskInv KD / ElasticFace-Arc checkpoint when available.
