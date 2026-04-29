# Adaptive Fusion Probe Conclusion

Recommendation: PROMISING SIGNAL

- Best fusion policy on masked-unmasked: mask_presence_gated_blackout
- Masked-unmasked ROC-AUC baseline: 0.9275
- Masked-unmasked ROC-AUC best policy: 0.9408
- Masked-unmasked gain vs baseline: 0.0133
- Unmasked-unmasked ROC-AUC baseline: 0.9544
- Unmasked-unmasked ROC-AUC best policy: 0.9544
- Unmasked regression vs baseline: 0.0000

Interpretation: this probe only tests whether a calibrated score policy has
signal using already-computed unmasked-recognizer variants. It still needs a
real mask-aware model for the final comparison.

Best policy includes unmasked-unmasked evaluation.
