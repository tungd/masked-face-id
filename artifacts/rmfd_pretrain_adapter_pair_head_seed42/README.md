# RMFD Masked Pretraining + Pair Head Seed 42

This artifact records a probe for pretraining a mask-aware residual adapter on
all available masked RMFD images, then fine-tuning/evaluating the pair head on
the same held-out paired-identity split used by the full RMFD pair-head run.

Run configuration:

- Runtime: Colab GPU, NVIDIA L4
- Script: `scripts/probe_rmfd_pretrain_adapter_pair_head.py`
- Commit at run time: `c311043`
- Data root: `/content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset`
- Seed: 42
- Train identities: 300
- Eval identities: 100
- Max images per condition: 8
- Train pairs per case: 10,000
- Eval pairs per case: 2,000
- Adapter pretraining epochs: 40
- Pair-head epochs: 60 for both raw and adapted conditions

Pretraining setup:

- Discovered masked images: 2,203
- Embedded masked images for pretraining: 1,097
- Embedding records requested overall: 5,403
- Embedding records embedded overall: 4,197
- Embedding failures overall: 1,206

Result:

| Model | Masked-unmasked ROC-AUC | Accuracy |
|---|---:|---:|
| Raw FaceNet baseline | 0.7999 | 0.7257 |
| Pair head only | 0.8214 | 0.7453 |
| Pretrained adapter + pair head | 0.7368 | 0.6814 |

The simple self-supervised residual adapter is not helpful. It reduced
masked-unmasked ROC-AUC by `0.0846` compared with the pair-head-only control,
although the masked-only policy still preserved unmasked-unmasked ROC-AUC by
bypassing that case to raw FaceNet.

Important dataset correction: this RMFD archive has `90,468` unmasked images
and `2,203` masked images, not 92k masked images. The pretraining stage used all
discovered masked images, but the usable masked pretraining pool is therefore
small.
