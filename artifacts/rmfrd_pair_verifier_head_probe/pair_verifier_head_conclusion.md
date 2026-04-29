# Pair Verifier Head Probe Conclusion

Recommendation: PROMISING

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best practical candidate: `pair_head_masked_cases_only`
- Best masked-unmasked ROC-AUC: `0.8057`
- Best gain vs baseline: `+0.0604`
- Baseline unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked regression vs baseline: `0.0000`

This is the strongest candidate so far. The all-cases pair head reaches the
same masked-unmasked ROC-AUC, but it lowers unmasked-unmasked ROC-AUC to
`0.9543`. The masked-only policy is therefore preferable: use the learned
pair verifier only when a mask is involved, and bypass unmasked-unmasked pairs
to the original FaceNet verifier.

The method freezes FaceNet and trains only a pair-level MLP over multi-view
embedding features. That keeps the contribution smaller and easier to defend
than full recognizer retraining while still adapting the verification decision
to masked faces.
