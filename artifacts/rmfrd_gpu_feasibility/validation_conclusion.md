# Validation Spike Conclusion

Recommendation: NOT YET FEASIBLE

- Question: can lightweight occlusion handling be a defensible course-project direction without retraining?
- Best lightweight model: lower_blackout_facenet_vggface2
- Masked-unmasked ROC-AUC baseline: 0.9275
- Masked-unmasked ROC-AUC upper-face: 0.8776
- Masked-unmasked ROC-AUC best lightweight: 0.9408
- Masked-unmasked gain vs baseline: 0.0133
- Unmasked-unmasked ROC-AUC baseline: 0.9544
- Unmasked-unmasked ROC-AUC best lightweight: 0.9117
- Unmasked regression vs baseline: 0.0426

Feasibility rule: promising only if a lightweight method improves masked-unmasked ROC-AUC by at least 0.02 while dropping unmasked-unmasked ROC-AUC by no more than 0.03.

Third model note: default candidate is FaceNet CASIA-WebFace; replace with MaskInv KD / ElasticFace-Arc checkpoint when available.
