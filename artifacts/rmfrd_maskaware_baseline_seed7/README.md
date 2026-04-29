# RMFRD Dedicated Mask-Aware Baseline Seed 7

This artifact records a Colab L4 run of `scripts/probe_maskaware_baseline.py`
on the seed-7 RMFRD split.

Configuration:

- Official repository: `fdbtrs/Masked-Face-Recognition-KD`
- Checkpoints: `MaskInvHG`, `MaskInvLG`, `ElasticFaceArcAug`
- Alignment: MTCNN landmarks with standard 112x112 ArcFace 5-point template
- FaceNet baseline: existing FaceNet/MTCNN full-face embedding path
- Eval identities: 80
- Eval images requested: 899
- Aligned images used: 762
- Detector failures: 137

Result: the best dedicated checkpoint is `elasticface_arc_aug`, with
masked-unmasked ROC-AUC `0.8724`. This beats the pair-head seed-7 result
`0.8238` by `+0.0486`, while keeping unmasked-unmasked regression versus
FaceNet to `0.0031`.
