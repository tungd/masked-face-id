# Dedicated Mask-Aware Baseline Conclusion

Recommendation: PROMISING BASELINE

- Seed: `7`
- Baseline FaceNet masked-unmasked ROC-AUC: `0.7965`
- Best dedicated checkpoint: `elasticface_arc_aug`
- Best dedicated masked-unmasked ROC-AUC: `0.8724`
- Best gain vs FaceNet baseline: `+0.0759`
- Baseline FaceNet unmasked-unmasked ROC-AUC: `0.9671`
- Best dedicated unmasked-unmasked ROC-AUC: `0.9640`
- Best dedicated unmasked regression vs FaceNet: `0.0031`
- Pair-head masked-unmasked ROC-AUC on same seed artifact: `0.8238`
- Best mask-aware gap vs pair head: `+0.0486`

This probe evaluates official MaskInv-family IResNet-100 checkpoints from
`fdbtrs/Masked-Face-Recognition-KD` as dedicated mask-aware baselines on the
same RMFRD identity split protocol.
