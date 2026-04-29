# Dedicated Mask-Aware Baseline Conclusion

Recommendation: PROMISING BASELINE

- Seed: `42`
- Baseline FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best dedicated checkpoint: `maskinv_lg`
- Best dedicated masked-unmasked ROC-AUC: `0.8273`
- Best gain vs FaceNet baseline: `+0.0820`
- Baseline FaceNet unmasked-unmasked ROC-AUC: `0.9668`
- Best dedicated unmasked-unmasked ROC-AUC: `0.9582`
- Best dedicated unmasked regression vs FaceNet: `0.0086`
- Pair-head masked-unmasked ROC-AUC on same seed artifact: `0.8057`
- Best mask-aware gap vs pair head: `+0.0216`

This probe evaluates official MaskInv-family IResNet-100 checkpoints from
`fdbtrs/Masked-Face-Recognition-KD` as dedicated mask-aware baselines on the
same RMFRD identity split protocol.
