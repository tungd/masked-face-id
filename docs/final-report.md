# Lightweight Pair-Head Adaptation for Masked Face Verification

## Abstract

This project studies a practical masked-face verification setting: an existing
unmasked face recognizer is already deployed, and we want to improve masked
verification without retraining or replacing the recognizer. The final approach
freezes FaceNet/InceptionResnetV1 and trains a small pair-level verifier head
over multi-view embedding features. The dedicated mask-aware recognizer is used
as a ceiling, not as the method we claim to beat.

On held-out RMFRD identity splits, the pair-head improves masked-unmasked
ROC-AUC from `0.7453` to `0.8057` on seed 42 and from `0.7965` to `0.8238` on
seed 7, while preserving unmasked-unmasked ROC-AUC by bypassing those pairs to
the original FaceNet verifier. Dedicated MaskInv/ElasticFace mask-aware models
remain stronger, reaching `0.8273` and `0.8724` masked-unmasked ROC-AUC on the
same two splits.

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

Splits:

- train identities: `140`
- evaluation identities: `80`
- max images per condition: `8`
- identity sets are disjoint between training and evaluation.

Pair cases:

- `masked-masked`,
- `masked-unmasked`,
- `unmasked-unmasked`.

Metric: ROC-AUC by pair case. Accuracy, threshold, FAR, and FRR are also saved
in artifacts, but ROC-AUC is the main comparison metric because it separates
ranking quality from threshold selection.

## Results

### Pair-Head Adaptation

| Seed | Model | Masked-unmasked ROC-AUC | Gain vs FaceNet | Unmasked-unmasked ROC-AUC |
|---:|---|---:|---:|---:|
| 42 | FaceNet baseline | 0.7453 | - | 0.9668 |
| 42 | Pair head, masked-only | 0.8057 | +0.0604 | 0.9668 |
| 7 | FaceNet baseline | 0.7965 | - | 0.9671 |
| 7 | Pair head, masked-only | 0.8238 | +0.0273 | 0.9671 |

The pair-head signal survives a second identity split. The gain is larger on
seed 42 than seed 7, but both are positive on the main masked-unmasked case.
The masked-only bypass keeps unmasked-unmasked ROC-AUC unchanged.

### Dedicated Mask-Aware Ceiling

| Seed | Best dedicated model | Masked-unmasked ROC-AUC | Gain vs FaceNet | Gap vs pair head |
|---:|---|---:|---:|---:|
| 42 | MaskInv-LG | 0.8273 | +0.0820 | +0.0216 |
| 7 | ElasticFace-Arc-Aug | 0.8724 | +0.0759 | +0.0486 |

The dedicated mask-aware recognizer is the expected ceiling. It is trained for
masked face recognition and uses an IResNet-100-style backbone, so it is not a
fair deployment-cost match for the small pair-head adapter. The useful claim is
therefore not "the pair head beats a dedicated model"; the useful claim is that
a lightweight adapter recovers a meaningful portion of the masked-unmasked gap
without replacing the deployed recognizer.

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
- larger LFW synthetic-mask training with real RMFRD evaluation.

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
target dataset distribution. It also trails the dedicated mask-aware model by
`0.0216` to `0.0486` ROC-AUC on masked-unmasked pairs, so it should be framed as
a lightweight adaptation strategy rather than a state-of-the-art recognizer.

## Reproducibility

Primary scripts:

- `scripts/probe_pair_verifier_head.py`
- `scripts/probe_maskaware_baseline.py`
- `scripts/probe_pair_head_synthetic_train_real_eval.py`
- `scripts/create_lfw_synthetic_mask_pairs.py`
- `scripts/install_colab_deps.py`

Main artifacts:

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
