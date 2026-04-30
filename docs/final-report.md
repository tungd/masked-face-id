# Lightweight Pair-Head Adaptation for Masked Face Verification

## Abstract

This project studies a practical masked-face verification setting: an existing
unmasked face recognizer is already deployed, and we want to improve masked
verification without retraining or replacing the recognizer. The final approach
freezes FaceNet/InceptionResnetV1 and trains a small pair-level verifier head
over multi-view embedding features. The dedicated mask-aware recognizer is used
as a ceiling, not as the method we claim to beat.

On held-out RMFD/RMFRD paired identity splits, the pair-head improves
masked-unmasked ROC-AUC from `0.7972 +/- 0.0027` to `0.8228 +/- 0.0158` across
three seeds. The masked-only bypass preserves unmasked-unmasked ROC-AUC at
`0.9694 +/- 0.0047`. Feature ablations show that dense pair interactions drive
the gain; cosine scores alone are close to the baseline. A stronger public
ArcFace/InsightFace comparison was also tested and was a useful negative
control: direct-crop `buffalo_l` reached only `0.7474` masked-unmasked ROC-AUC
on the seed-42 split, and its simple pair head hurt performance.

## Problem

Face recognizers trained mostly on unmasked faces can fail when one image is
masked and the other is not. This is the important verification case for access
control or identity matching against legacy galleries: a masked probe must match
an unmasked reference.

The project question is:

> Can a small adaptation layer improve masked verification for an existing
> unmasked recognizer, while preserving its behavior on unmasked pairs?

## Final Method

The recognizer is kept frozen. For each image, the pipeline extracts five
FaceNet embeddings from aligned face views:

- full face,
- lower-face blackout,
- lower-face blur,
- upper-only,
- eye-band.

For a candidate pair, the verifier head receives:

- one-hot pair type,
- cosine similarities for the five views,
- score summary statistics,
- dense pair interactions for each view: absolute embedding difference and
  elementwise embedding product.

The trainable component is a small MLP binary verifier. It predicts match
probability for pair-level examples sampled from training identities. At
inference, the practical policy is:

- use the pair head for `masked-masked` and `masked-unmasked` pairs,
- use raw FaceNet cosine similarity for `unmasked-unmasked` pairs.

That masked-only policy is important because it preserves the existing
recognizer where masks are not involved.

## Experimental Protocol

Dataset: RMFRD from the Real-World Masked Face Dataset family. The probe uses
identities that have both masked and unmasked examples.

The current paired subset scan found `403` usable identities with both masked
and unmasked images. The main validation uses three disjoint identity splits:

- seeds: `42`, `7`, `99`
- train identities per seed: `300`
- evaluation identities per seed: `100`
- max images per condition: `8`
- train pairs per case: `10000`
- evaluation pairs per case: `2000`
- calibration split: `20%` of training pairs for threshold selection.

Pair cases:

- `masked-masked`,
- `masked-unmasked`,
- `unmasked-unmasked`.

Metric: ROC-AUC by pair case. Accuracy, threshold, FAR, FRR, and TAR are also
saved in artifacts, but ROC-AUC is the main comparison metric because it
separates ranking quality from threshold selection. Threshold analysis uses
calibration pairs only for threshold choice and applies those thresholds to
held-out eval pairs.

## Results

### Pair-Head Robustness

| Seed | FaceNet masked-unmasked | Pair head masked-only | Gain | Unmasked-unmasked preserved |
|---:|---:|---:|---:|---:|
| 7 | 0.7945 | 0.8077 | +0.0132 | 0.9651 |
| 42 | 0.7999 | 0.8214 | +0.0215 | 0.9687 |
| 99 | 0.7973 | 0.8393 | +0.0420 | 0.9744 |
| Mean +/- std | 0.7972 +/- 0.0027 | 0.8228 +/- 0.0158 | +0.0256 | 0.9694 +/- 0.0047 |

The harder validation keeps the original conclusion but makes it more modest:
the gain is consistent across three held-out identity splits, but it is not a
large margin. The masked-only bypass is still important because the all-cases
head slightly degrades unmasked-unmasked ROC-AUC (`0.9586 +/- 0.0049`), while
the bypass preserves the raw FaceNet value.

### Feature Ablation

| Feature set | Masked-unmasked ROC-AUC mean +/- std |
|---|---:|
| Full five-view features + dense interactions | 0.8228 +/- 0.0158 |
| Full-face dense interactions only | 0.8185 +/- 0.0205 |
| Dense interactions only, all views | 0.8062 +/- 0.0161 |
| Cosine scores only | 0.7986 +/- 0.0021 |
| Cosine scores + score statistics | 0.7969 +/- 0.0045 |
| Raw FaceNet baseline | 0.7972 +/- 0.0027 |

This ablation says the pair head is not winning because of a simple learned
threshold over cosine scores. The useful signal is mostly in the dense pair
interactions: absolute embedding differences and elementwise products.

### Threshold Calibration

| Model | Eval FAR mean | Eval TAR mean | Eval accuracy mean |
|---|---:|---:|---:|
| FaceNet baseline | 0.0481 | 0.3866 | 0.6638 |
| Pair head, cosine-only | 0.0495 | 0.3999 | 0.6697 |
| Pair head, full features | 0.0502 | 0.3769 | 0.6577 |

Thresholds were chosen on calibration pairs for nominal FAR `0.05` and then
applied to held-out eval pairs. The full pair head improves ranking ROC-AUC, but
it does not improve this fixed operating point. This is a useful limitation for
the final argument: the method is defensible as a ranking/verifier adaptation,
but deployment threshold calibration would need more work.

### Stronger Public Recognizer Check

| Model | Masked-masked ROC-AUC | Masked-unmasked ROC-AUC | Unmasked-unmasked ROC-AUC |
|---|---:|---:|---:|
| InsightFace `buffalo_l` raw cosine | 0.7995 | 0.7474 | 0.9383 |
| InsightFace `buffalo_l` pair head | 0.7766 | 0.7186 | 0.9383 |
| FaceNet raw cosine, same seed | 0.8578 | 0.7999 | 0.9687 |
| FaceNet pair head, same seed | 0.8910 | 0.8214 | 0.9687 |

This comparison did not disprove the FaceNet pair-head result. It showed an
integration caveat: InsightFace's detector found only `21` of `4637` selected
RMFD crops, so the probe had to fall back to direct square-padded ArcFace crops
for the rest. Under that setup, `buffalo_l` is weaker than the FaceNet/MTCNN
pipeline on this dataset, and a simple full-embedding pair head is harmful.

### Dedicated Mask-Aware Ceiling

Earlier ceiling runs with dedicated MaskInv/ElasticFace-style models remain
useful context, but they are not the deployment-cost comparison we claim to
beat. On the earlier two-seed protocol, the best dedicated models reached
`0.8273` and `0.8724` masked-unmasked ROC-AUC. They are trained for masked face
recognition and use stronger IResNet-style backbones, so the right framing is:
the pair head recovers part of the masked-unmasked gap without replacing the
deployed recognizer.

### Larger Synthetic-Training Extension

After the main result, we also ran a clean Colab extension that trains the pair
head on a larger synthetic masked/unmasked LFW root and evaluates on real RMFRD
identities. This used the reproducible LFW fallback because the clean runtime
did not have Kaggle credentials or a mounted official SMFRD archive. The RMFRD
evaluation identity set is separate from the two main RMFRD splits, so the
extension is a transfer check rather than a row-by-row comparison to the main
result.

| Train data | Eval data | FaceNet masked-unmasked | Pair head masked-only | Gain | Unmasked-unmasked |
|---|---|---:|---:|---:|---:|
| LFW synthetic, 1,000 train identities | RMFRD, 80 eval identities | 0.7900 | 0.7914 | +0.0014 | 0.9570 |

This is an important negative transfer result. The gain is technically
positive, but far smaller than the RMFRD-trained pair head gains above. It
suggests that simple synthetic lower-face masks alone do not transfer enough to
replace target-domain pair supervision. The official SMFRD archive remains a
reasonable future dataset upgrade, but this fallback run should not become the
main project claim.

## Negative Probes

Several alternatives were explored and rejected:

- fixed score fusion,
- logistic-regression score fusion,
- lower-face blackout and blur gates,
- frozen residual/ridge/mean-shift embedding adapters,
- test-time occlusion ensembles,
- contrastive residual embedding adapter,
- partial fine-tuning of the FaceNet tail,
- scratch-trained periocular specialist using MediaPipe face landmarks,
- larger LFW synthetic-mask training with real RMFRD evaluation,
- residual adapter pretraining on all available masked RMFD images,
- InsightFace/ArcFace full-embedding pair head.

The periocular specialist is especially useful as a negative result. It trained
successfully, but on seed 42 held-out identities it reached only `0.6358`
masked-unmasked ROC-AUC, and fusing it with FaceNet hurt performance. This
supports the final choice: pair-level evidence from a strong frozen recognizer
is more reliable than a small crop-only recognizer trained from scratch.

## Interpretation

The pair head works because it does not ask the frozen recognizer to produce a
single perfect masked embedding. Instead, it gives the verifier several
occlusion views and lets it learn how score patterns differ for genuine and
impostor pairs. This is a better fit for the course project than full
recognizer retraining:

- the method is small,
- it is recognizer-agnostic in principle,
- it directly targets verification,
- it preserves the legacy unmasked path.

The main limitation is that the pair head still learns from pair labels on the
target dataset distribution. The calibration analysis also shows that better
ROC-AUC does not automatically produce a better fixed-FAR operating point. It
should be framed as a lightweight adaptation strategy rather than a
state-of-the-art recognizer.

## Reproducibility

Primary scripts:

- `scripts/probe_pair_verifier_head.py`
- `scripts/probe_pair_head_robustness.py`
- `scripts/probe_insightface_pair_head.py`
- `scripts/probe_maskaware_baseline.py`
- `scripts/probe_rmfd_pretrain_adapter_pair_head.py`
- `scripts/scan_rmfd_paired_identities.py`
- `scripts/probe_pair_head_synthetic_train_real_eval.py`
- `scripts/create_lfw_synthetic_mask_pairs.py`
- `scripts/install_colab_deps.py`

Main artifacts:

- `artifacts/rmfd_pair_head_robustness_seed42_7_99/`
- `artifacts/insightface_pair_head_seed42/`
- `artifacts/rmfd_pair_verifier_head_full_seed42/`
- `artifacts/rmfd_pretrain_adapter_pair_head_seed42/`
- `artifacts/rmfd_paired_identity_scan/`
- `artifacts/rmfrd_pair_verifier_head_probe/`
- `artifacts/rmfrd_pair_verifier_head_seed7/`
- `artifacts/rmfrd_maskaware_baseline_seed42/`
- `artifacts/rmfrd_maskaware_baseline_seed7/`
- `artifacts/lfw_synthetic_train_rmfrd_eval_seed42/`

Colab notebook:

- `notebooks/masked_face_pair_head_final.ipynb`
- `notebooks/smfrd_pair_head_extension.ipynb`

Presentation:

- `slides/pair_head_final_presentation.html`

## References

- Schroff, Kalenichenko, and Philbin. FaceNet: A Unified Embedding for Face
  Recognition and Clustering. CVPR 2015. https://arxiv.org/abs/1503.03832
- Zhang, Zhang, Li, and Qiao. Joint Face Detection and Alignment using
  Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters
  2016. https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- Huber, Boutros, Kirchbuchner, and Damer. Mask-invariant Face Recognition
  through Template-level Knowledge Distillation. FG 2021.
  https://github.com/fdbtrs/Masked-Face-Recognition-KD
- Real-World Masked Face Dataset / RMFRD dataset summary.
  https://hyper.ai/en/datasets/19431
