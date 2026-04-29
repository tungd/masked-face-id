# RMFRD Training Adaptation Probe

This artifact records three Colab L4 runs of `scripts/probe_training_adaptation.py`
on the self-built RMFRD split.

Common run parameters:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Train identities: 140
- Eval identities: 80
- Max images per condition: 8
- Pairs per case: 800
- Device: CUDA, Colab L4
- Aligned images used: 2125
- Detector failures: 457

Methods:

- `contrastive_adapter_masked_only`: residual MLP adapter trained on frozen
  FaceNet embeddings against unmasked identity templates.
- `partial_finetune_full`: FaceNet tail fine-tuned with supervised contrastive
  identity batches, optionally with frozen-embedding distillation.

Best result:

- Configuration: large tail distillation
- Trainable prefixes: `repeat_3,block8,last_linear,last_bn`
- Fine-tune steps: 3000
- Fine-tune LR: `1e-5`
- Distillation: all samples `0.05`, unmasked samples `1.0`
- Masked-unmasked ROC-AUC: `0.7721`
- Gain vs baseline: `+0.0267`
- Unmasked-unmasked ROC-AUC: `0.9515`
- Unmasked regression: `0.0153`

Interpretation: partial fine-tuning is the first direction here that clears the
original feasibility rule on this held-out split. The contrastive adapter is
not promising as implemented.
