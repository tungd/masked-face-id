# Occlusion Fusion Feasibility Spike

## Goal

Test whether a course-project framing around lightweight, model-agnostic mask handling is defensible, even if a dedicated masked-face model remains the expected upper bound.

## Research Question

Can simple occlusion-aware preprocessing or score fusion recover any masked-vs-unmasked verification performance without retraining the recognizer and without badly degrading unmasked-vs-unmasked verification?

## Variants

- Full-face FaceNet baseline.
- Hard upper-face crop.
- Lower-face blackout on the detected face tensor.
- Lower-face blur on the detected face tensor.
- Score fusion between full-face and each lightweight occlusion variant.
- Masked-specific candidate model as an upper-bound slot, not as the main contribution.

## Evaluation

- Use RMFRD/SMFRD or PKU-style real masked/unmasked identities.
- Build masked-masked, masked-unmasked, and unmasked-unmasked verification pairs.
- Tune fusion alpha only on masked-unmasked calibration pairs.
- Report held-out metrics for all cases.
- Use ROC-AUC, accuracy, FAR, and FRR.

## Feasibility Bar

The project direction is promising if at least one lightweight method improves masked-unmasked ROC-AUC by 0.02 or more while degrading unmasked-unmasked ROC-AUC by no more than 0.03.

If no lightweight method clears that bar, the project can still be framed as a negative result and failure-mode study, but it is weaker as a constructive method project.
