# Occlusion Ensemble Probe Conclusion

Recommendation: NO FULL-COVERAGE GAIN

- Baseline full FaceNet masked-unmasked ROC-AUC: 0.7646
- Best full-coverage ensemble on masked-unmasked: `ensemble_top2_safe`
- Best full-coverage ensemble masked-unmasked ROC-AUC: 0.7629
- Full-coverage ensemble gain vs baseline: -0.0017
- Best 80% coverage policy: `gated_upper_only`
- Best 80% coverage masked-unmasked ROC-AUC: 0.7807
- Best 80% coverage ensemble: `ensemble_disagreement_penalty_0.50`
- Best 80% coverage ensemble masked-unmasked ROC-AUC: 0.7729
- Unmasked-unmasked ROC-AUC for gated/ensemble policies: 0.9710

Full-coverage occlusion ensembling does not beat the full-face baseline on this
split. The useful signal is selective verification: view-disagreement ranking
improves masked-unmasked AUC when the system is allowed to abstain on the least
reliable 20% of pairs. This supports a reliability-aware project framing more
than a pure accuracy-improvement framing.
