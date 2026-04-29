# ArcFace Fine-Tune Probe Conclusion

Recommendation: NOT YET PROMISING

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best ArcFace candidate: `arcface_finetune_masked_pairs_only`
- Best masked-unmasked ROC-AUC: `0.6922`
- Best gain vs baseline: `-0.0531`
- Baseline unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked regression vs baseline: `0.0000`

This probe fine-tunes the FaceNet tail with an ArcFace identity-classification
objective and frozen-embedding distillation. Evaluation identities are disjoint
from training identities.

The likely failure mode is identity-classification overfitting: the ArcFace
training loss falls sharply, but the held-out masked-unmasked geometry gets
worse. This makes it weaker than the pair verifier head and the earlier
distilled supervised-contrastive partial fine-tune.
