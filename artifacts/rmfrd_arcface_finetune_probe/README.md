# RMFRD ArcFace Fine-Tune Probe

This artifact records a Colab L4 run of `scripts/probe_arcface_finetune.py` on
the self-built RMFRD split.

Configuration:

- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Train identities requested: 140
- Eval identities requested: 80
- Aligned images used: 2125
- Detector failures: 457
- Trainable FaceNet prefixes: `repeat_3`, `block8`, `last_linear`, `last_bn`
- Trainable FaceNet parameters: 10523136
- ArcFace classes after alignment: 108
- Steps: 3000
- LR: `1e-5`
- Distillation: all samples `0.05`, unmasked samples `1.0`

Result: this is not promising as configured. Masked-unmasked ROC-AUC fell from
`0.7453` to `0.6922`, even with unmasked-unmasked preserved by bypass.
