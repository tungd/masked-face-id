# Adaptive Fusion vs Mask-Aware Recognizer Spike

## Goal

Evaluate a course-project idea where the contribution is an adaptive fusion
layer that improves an existing unmasked face recognizer for masked-face
verification, then compare it against a dedicated mask-aware recognizer.

## Project Claim

An existing unmasked-only recognizer can be made more useful for masked
verification by fusing full-face and occlusion-focused scores with a small
calibrated policy. The fused system should be compared against, not expected to
dominate, a dedicated mask-aware recognizer.

## Baselines

- **Unmasked recognizer baseline:** FaceNet VGGFace2 full-face embedding.
- **Naive occlusion baselines:** upper-face crop, lower-face blackout,
  lower-face blur.
- **Dedicated mask-aware baseline:** MaskInv KD / ElasticFace-Arc or another
  public masked-face-recognition checkpoint.

## Proposed Method

Train or tune a lightweight fusion policy over scores from the unmasked
recognizer variants.

Initial feature candidates:

- Verification case: masked-masked, masked-unmasked, unmasked-unmasked.
- Full-face cosine score.
- Upper-face cosine score.
- Lower-blackout cosine score.
- Lower-blur cosine score.
- Score gaps such as full minus upper and full minus blackout.
- Optional mask/occlusion confidence if a detector is added.

Initial policy candidates:

- Case-specific alpha fusion.
- Logistic regression over score features.
- Small calibrated decision model trained only on calibration pairs.

## Evaluation Protocol

- Use RMFRD/SMFRD or PKU-style real masked/unmasked identities.
- Build masked-masked, masked-unmasked, and unmasked-unmasked verification
  pairs.
- Split pairs into calibration and held-out evaluation partitions by identity or
  stable pair hash.
- Tune fusion policy only on calibration data.
- Report held-out metrics for every model and verification case.
- Compare against a real mask-aware model, not the CASIA FaceNet placeholder.

## Success Criteria

The project is defensible if the fusion policy:

- Improves masked-unmasked verification over the unmasked full-face baseline.
- Preserves unmasked-unmasked verification better than raw occlusion-only
  preprocessing.
- Occupies a meaningful middle ground against the dedicated mask-aware model:
  simpler and model-agnostic, possibly less accurate, but empirically useful.

The project does not require beating the dedicated mask-aware model. The key
argument is whether the fusion policy is a practical adaptation layer for an
existing recognizer.

## Immediate Implementation Gap

The current notebook has only:

- raw preprocessing variants,
- fixed alpha score fusion, and
- a CASIA FaceNet placeholder for the mask-aware slot.

Next implementation work:

1. Replace the CASIA placeholder with a real mask-aware checkpoint.
2. Add an adaptive fusion policy, starting with logistic regression over score
   features.
3. Re-run the GPU RMFRD evaluation and report fused-vs-unmasked-vs-mask-aware
   results.
