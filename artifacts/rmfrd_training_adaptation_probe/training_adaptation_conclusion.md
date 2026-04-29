# Training Adaptation Probe Conclusion

Recommendation: PROMISING for partial fine-tuning with distillation.

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best trained method: `partial_finetune_full`
- Best configuration: large tail plus frozen-embedding distillation
- Best masked-unmasked ROC-AUC: `0.7721`
- Gain vs baseline: `+0.0267`
- Baseline unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked-unmasked ROC-AUC: `0.9515`
- Unmasked regression vs baseline: `0.0153`

The pure contrastive embedding adapter was not useful in these runs. It
consistently reduced masked-unmasked ROC-AUC to about `0.706`, despite low
training loss, which suggests calibration-identity overfitting.

The unregularized partial fine-tune improved masked-unmasked ROC-AUC but caused
too much unmasked-unmasked regression. The useful result came from unfreezing
`repeat_3`, `block8`, `last_linear`, and `last_bn`, then adding a small
all-sample distillation term plus a stronger unmasked distillation term. That
kept the recognizer close enough to the original embedding space while allowing
masked embeddings to move.
