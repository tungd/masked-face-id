# Negative Results Summary

These probes are part of the final story because they show why the project
settled on a frozen FaceNet pair-head verifier instead of a broader model
search.

| Probe | Outcome | Use in final story |
|---|---|---|
| Upper-face preprocessing | Hurt or failed to improve masked-unmasked ranking | Naive occlusion removal is insufficient |
| Fixed/gated occlusion views | Weak or inconsistent gains | Simple score policies are not enough |
| Frozen residual/ridge/mean-shift adapters | Small or unstable gains | Embedding-level correction was limited |
| Test-time occlusion ensembles | Did not become a robust policy | More views alone do not solve calibration |
| Contrastive residual adapter | Added complexity without a strong result | Kept the final method smaller |
| Partial FaceNet tail fine-tuning | More risk and not the frozen-deployment story | Future work, not the final claim |
| Periocular specialist | Seed-42 masked-unmasked ROC-AUC only `0.6358`; fusion hurt FaceNet | Crop-only training was not robust |
| Larger LFW synthetic-mask training | Real RMFRD masked-unmasked gain only `+0.0014` ROC-AUC | Synthetic masks did not transfer enough |
| RMFD residual adapter pretraining | Hurt the pair-head result | More target images were not automatically useful |
| InsightFace pair head | `buffalo_l` direct-crop baseline reached `0.7474` masked-unmasked ROC-AUC and the simple pair head fell to `0.7186` | Detector/crop integration details matter |
| Dedicated mask-aware recognizer | Stronger masked-unmasked ceiling around `0.827` to `0.872` ROC-AUC in earlier runs | Honest upper-bound comparison |
| Pair-head dense interactions | Best lightweight method: `0.8228 +/- 0.0158` masked-unmasked ROC-AUC | Main contribution |

The final claim should stay narrow: a lightweight pair-level verifier improves
masked-unmasked ranking for the frozen FaceNet pipeline and preserves the
unmasked path, but it does not become a production biometric system or beat a
dedicated mask-aware recognizer.
