# RMFRD Dedicated Mask-Aware Baseline Seed 42

This artifact records a Colab L4 run of `scripts/probe_maskaware_baseline.py`
on the seed-42 RMFRD split.

Configuration:

- Official repository: `fdbtrs/Masked-Face-Recognition-KD`
- Checkpoints: `MaskInvHG`, `MaskInvLG`, `ElasticFaceArcAug`
- Alignment: MTCNN landmarks with standard 112x112 ArcFace 5-point template
- FaceNet baseline: existing FaceNet/MTCNN full-face embedding path
- Eval identities: 80
- Eval images requested: 952
- Aligned images used: 785
- Detector failures: 167

Result: the best dedicated checkpoint is `maskinv_lg`, with masked-unmasked
ROC-AUC `0.8273`. This beats the pair-head seed-42 result `0.8057` by
`+0.0216`, while keeping unmasked-unmasked regression versus FaceNet to
`0.0086`.
