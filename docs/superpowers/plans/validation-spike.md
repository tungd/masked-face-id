# Validation Spike Plan

## Goal

Validate whether a simple occlusion-aware face verification approach is worth pursuing before investing in a full project.

## Hypothesis

A recognizer that ignores or downweights masked lower-face regions can improve masked-vs-unmasked verification relative to a plain baseline, without causing a large regression on unmasked verification.

## Scope

- Do not use MaskedFace-Net as the main evaluation benchmark.
- Use a real masked-recognition dataset instead.
- Preferred dataset: PKU-Masked-Face.
- Acceptable fallback: RMFRD/SMFRD.
- Start with a small slice, around 100-200 identities, not the full dataset.

## Execution Plan

1. Set up a Colab runtime and create or clone the working repo.
2. Download a small benchmark subset with both masked and unmasked images per identity.
3. Implement only three variants for the first pass:
   - baseline recognizer
   - simple upper-face or occlusion-aware preprocessing
   - one masked-face-specific model
4. Build verification pairs for three cases:
   - masked-masked
   - masked-unmasked
   - unmasked-unmasked
5. Evaluate with ROC-AUC, verification accuracy, and FAR/FRR.
6. Save a compact results table.
7. Collect a few qualitative examples where the methods clearly differ.
8. Make a go/no-go recommendation from the results.

## Deliverables

- One runnable notebook for the spike.
- One compact results table.
- One short conclusion with a clear recommendation.

## Go/No-Go Bar

- Go only if masked-unmasked verification improves materially and unmasked performance does not degrade badly.
- No-go if the gains are small, unstable, or only visible on synthetic masks.

## Practical Note

The GitHub repo is currently empty, so the first CLI step should be to create the repo or notebook scaffold before running the benchmark.
