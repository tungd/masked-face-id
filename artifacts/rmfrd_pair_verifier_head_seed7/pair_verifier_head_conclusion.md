# Pair Verifier Head Probe Conclusion

Recommendation: PROMISING

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7965`
- Best candidate: `pair_head_masked_cases_only`
- Best masked-unmasked ROC-AUC: `0.8238`
- Best gain vs baseline: `+0.0273`
- Baseline unmasked-unmasked ROC-AUC: `0.9671`
- Best unmasked-unmasked ROC-AUC: `0.9671`
- Best unmasked regression vs baseline: `0.0000`

This second identity split gives a smaller but still meaningful positive
result. It supports keeping the frozen pair verifier as the main direction,
while still requiring more seeds and a dedicated mask-aware baseline before
making a final claim.
