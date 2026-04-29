# Technique Options

## Current Backup

The backup method is a mask-presence gate:

- use full-face FaceNet for unmasked-unmasked pairs,
- use lower-face blackout for pairs involving masks.

On the RMFRD L4 probe this improved masked-unmasked ROC-AUC from `0.9275` to
`0.9408` while preserving unmasked-unmasked ROC-AUC at `0.9544`. The gain is
small, but it is simple and defensible as a baseline.

## Current Main Candidate: Frozen-Recognizer Pair Verifier

Train a small pair verifier on top of a frozen unmasked recognizer.

Idea:

- Extract FaceNet embeddings for masked and unmasked images across multiple
  test-time views.
- Freeze FaceNet.
- Train a small MLP over pair-level features, including per-view similarities
  and dense embedding interactions.
- Use the learned verifier only for pairs involving masks.
- Bypass unmasked-unmasked pairs to the original FaceNet verifier.
- Evaluate verification with:
  - raw FaceNet,
  - mask-presence gate,
  - learned pair verifier,
  - dedicated mask-aware recognizer.

Why this is promising:

- It directly targets the masked-to-unmasked mismatch.
- It is a small course-project-sized adaptation layer rather than a full
  recognizer retrain.
- It does not require retraining the full face recognizer.
- It is more original than hand-tuned score fusion and stronger than the
  frozen embedding adapter on the first held-out split.

Probe:

1. Save per-image multi-view embeddings from the notebook.
2. Build held-out identity splits.
3. Train a pair-level MLP on calibration pairs.
4. Evaluate masked-unmasked and unmasked-unmasked pairs on held-out identities.

Success signal:

- masked-unmasked ROC-AUC improves over full-face baseline,
- unmasked-unmasked is preserved by bypassing the verifier for unmasked pairs,
- dedicated mask-aware model remains the expected upper bound.

## Option: Quality-Aware Score Fusion

Instead of a fixed alpha, learn alpha from quality signals:

- MTCNN confidence,
- face box size,
- embedding norm before normalization,
- score disagreement between full-face and occlusion variants,
- mask presence / pair type.

This may fix the current logistic-regression failure by predicting a fusion
weight rather than a final match probability.

Probe:

- Learn alpha on calibration pairs by optimizing held-out ROC-AUC or log loss.
- Constrain alpha so unmasked-unmasked pairs default to full-face baseline.

Risk:

- Pair count is small after detector skips, so the alpha model can overfit.

## Option: Case-Specific Calibration

Keep the raw full-face score, but calibrate thresholds or score transforms per
case:

- masked-masked,
- masked-unmasked,
- unmasked-unmasked.

This is operationally useful but weaker as a project contribution because it
does not create a better representation.

## Option: Periocular / Upper-Face Specialist

Train a small periocular head rather than resizing the upper crop into FaceNet.

The simple upper-face crop performed poorly, but a learned upper-face head could
still be meaningful. This aligns with the broader periocular-recognition
literature for masked faces.

Risk:

- Requires more training data and a clearer architecture.
- May drift away from the "adapt an existing recognizer" framing.

## Option: Synthetic Mask Augmentation

Apply synthetic masks to unmasked faces and train an adapter or classifier using
paired synthetic masked/unmasked examples.

This is useful if real masked pairs are limited. It also lets us create more
calibration data without changing the evaluation set.

Risk:

- The validation plan explicitly avoids relying only on synthetic masks, so this
  should be training augmentation only, not the benchmark.

## Option: Inpainting Before Recognition

Use a generative model to reconstruct the lower face before running the
unmasked recognizer.

This is probably too large and fragile for this project. It can be discussed as
related work, but it is not the best implementation path.

## Dedicated Mask-Aware Baseline

We still need a real mask-aware recognizer for the final comparison. The main
candidate is MaskInv KD / ElasticFace-Arc from the official
`fdbtrs/Masked-Face-Recognition-KD` repository.

The final story should not require beating this model. The project can argue
that a frozen-recognizer adapter is simpler, model-agnostic, and useful when an
organization already has an unmasked recognizer deployed.

## Recommended Next Probe

The pair verifier is the strongest current candidate, so the next probes should
test whether that signal is stable and whether a more recognizer-like training
objective can beat it.

Priority order:

- repeat the pair verifier over additional seeds or identity splits,
- add a dedicated mask-aware recognizer baseline,
- try ArcFace-style identity-classification fine-tuning,
- try dual-branch full-face plus periocular training.

If the pair verifier survives repeated splits and remains competitive with a
dedicated mask-aware recognizer, it is a defensible final project direction.

## Frozen-Recognizer Adapter Probe Result

The initial held-out RMFRD probe is a weak positive signal.

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7645`
- Best adapter: `mean_shift_full_masked_only`
- Best masked-unmasked ROC-AUC: `0.7775`
- Gain vs baseline: `+0.0130`
- Unmasked-unmasked ROC-AUC: unchanged at `0.9710`

The unconstrained ridge adapters improved masked-masked pairs but hurt
masked-unmasked pairs, suggesting that a free linear projection overfits the
calibration identities. The constrained mean-shift adapter is more useful for
the project framing because it is simple, frozen-recognizer compatible, and
does not touch unmasked-unmasked comparisons.

This is not enough for a final claim yet. The next evaluation should compare
the adapter against a dedicated mask-aware recognizer and repeat the adapter
probe across at least one additional split or seed.

## Test-Time Occlusion Ensemble Probe Result

The full-coverage ensemble idea did not beat the full-face baseline on the
held-out split.

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7646`
- Best full-coverage ensemble: `ensemble_top2_safe`
- Best full-coverage ensemble ROC-AUC: `0.7629`
- Gain vs baseline: `-0.0017`

The more interesting signal is reliability-aware selective verification. When
ranking pairs by occlusion-view disagreement and keeping only 80% coverage:

- Best policy: `gated_upper_only`
- Masked-unmasked ROC-AUC at 80% coverage: `0.7807`
- Best ensemble policy: `ensemble_disagreement_penalty_0.50`
- Ensemble masked-unmasked ROC-AUC at 80% coverage: `0.7729`

This suggests a possible project pivot: not "we always improve masked
recognition", but "we adapt an unmasked recognizer with mask-aware reliability
estimation and abstention." That is more original, but it needs careful
evaluation with coverage-risk curves and a dedicated mask-aware baseline.

## Training Adaptation Probe Result

The first actual training probe found a stronger direction than the frozen
adapter and test-time ensemble: partial fine-tuning with frozen-embedding
distillation.

Three configurations were tested on the same held-out RMFRD split:

- Unregularized large-tail fine-tune:
  - masked-unmasked ROC-AUC: `0.7606`
  - gain vs baseline: `+0.0153`
  - unmasked-unmasked regression: `0.0695`
- Small-tail fine-tune with distillation:
  - masked-unmasked ROC-AUC: `0.7419`
  - gain vs baseline: `-0.0034`
  - unmasked-unmasked regression: `0.0304`
- Large-tail fine-tune with distillation:
  - masked-unmasked ROC-AUC: `0.7721`
  - gain vs baseline: `+0.0267`
  - unmasked-unmasked regression: `0.0153`

The best configuration unfreezes `repeat_3`, `block8`, `last_linear`, and
`last_bn`, trains with supervised contrastive batches, and regularizes against
the frozen FaceNet embedding space. This clears the original feasibility rule
on this split.

The contrastive residual adapter did not work: it reduced masked-unmasked
ROC-AUC to about `0.706` even though training loss dropped quickly. The likely
failure mode is calibration-identity overfitting.

## Pair Verifier Head Probe Result

The frozen-recognizer pair verifier is the strongest current candidate.

Seed 42:

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best practical candidate: `pair_head_masked_cases_only`
- Best masked-unmasked ROC-AUC: `0.8057`
- Gain vs baseline: `+0.0604`
- Baseline unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked-unmasked ROC-AUC: `0.9668`
- Unmasked-unmasked regression: `0.0000`

Seed 7:

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7965`
- Best practical candidate: `pair_head_masked_cases_only`
- Best masked-unmasked ROC-AUC: `0.8238`
- Gain vs baseline: `+0.0273`
- Baseline unmasked-unmasked ROC-AUC: `0.9671`
- Best unmasked-unmasked ROC-AUC: `0.9671`
- Unmasked-unmasked regression: `0.0000`

The all-cases pair head reaches the same masked-unmasked ROC-AUC but regresses
unmasked-unmasked ROC-AUC to `0.9543`. The masked-only policy is preferable:
use the learned verifier only when a mask is involved, and send
unmasked-unmasked pairs through the original recognizer.

This is more promising than the partial fine-tuning probe: the gain is larger
on seed 42, still positive on seed 7, the recognizer remains frozen, and the
unmasked case is preserved by construction. The remaining risk is the gap to a
dedicated mask-aware recognizer.

## ArcFace Fine-Tune Probe Result

The ArcFace-style identity-classification fine-tune is not promising as
configured.

- Baseline full FaceNet masked-unmasked ROC-AUC: `0.7453`
- Best ArcFace candidate: `arcface_finetune_masked_pairs_only`
- Best masked-unmasked ROC-AUC: `0.6922`
- Gain vs baseline: `-0.0531`
- Baseline unmasked-unmasked ROC-AUC: `0.9668`
- Best unmasked-unmasked ROC-AUC: `0.9668`
- Unmasked-unmasked regression: `0.0000`

The training loss drops quickly, but held-out masked-unmasked verification gets
worse. The likely failure mode is identity-classification overfitting to the
training identities, even with frozen-embedding distillation. This result
demotes ArcFace fine-tuning below both the pair verifier and the distilled
supervised-contrastive partial fine-tune.
