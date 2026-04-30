# SMFRD Extension Plan

This optional extension tests whether a larger synthetic masked dataset can
train the pair-head adapter better, while keeping the evaluation on real masked
faces.

## Goal

Train:

- synthetic or larger masked/unmasked root, ideally SMFRD paired with the
  original unmasked source images.

Evaluate:

- RMFRD real masked/unmasked identities.

The claim remains conservative: synthetic data is used only for adapter
training; the reported test result should come from real masked faces.

## Clean Colab Setup

From a fresh Colab runtime:

```bash
git clone https://github.com/tungd/masked-face-id.git /content/masked-face-id
cd /content/masked-face-id
python scripts/install_colab_deps.py
```

Inspect the empty dataset root:

```bash
python scripts/setup_masked_datasets.py \
  --root /content/datasets \
  --summarize-only
```

## Dataset Sources

RMFRD has a Google Drive link in the official dataset README and a Kaggle mirror.
The setup helper supports both:

```bash
python scripts/setup_masked_datasets.py \
  --root /content/datasets \
  --rmfrd-source gdrive
```

or, with a configured Kaggle token:

```bash
python scripts/setup_masked_datasets.py \
  --root /content/datasets \
  --rmfrd-source kaggle
```

For SMFRD, the official README lists Baidu links for the simulated WebFace,
LFW, AgeDB-30, and CFP-FP sets. Kaggle has an LFW-SMFRD mirror. The helper can
download that mirror when Kaggle credentials are available:

```bash
python scripts/setup_masked_datasets.py \
  --root /content/datasets \
  --smfrd-source kaggle
```

If the simulated masked archive and original unmasked archive are mounted from
Drive, build a normalized paired view:

```bash
python scripts/setup_masked_datasets.py \
  --root /content/datasets \
  --make-paired-view \
  --masked-root /content/datasets/smfrd/kaggle/lfw_masked/lfw_train \
  --unmasked-root /content/datasets/lfw/original \
  --paired-view-root /content/datasets/normalized/lfw_smfrd_paired
```

The paired view has the layout expected by the probe:

```text
normalized/lfw_smfrd_paired/
  masked/<identity>/*.jpg
  unmasked/<identity>/*.jpg
```

## Run Synthetic-Train Real-Eval Probe

```bash
python scripts/probe_pair_head_synthetic_train_real_eval.py \
  --train-data-root /content/datasets/normalized/lfw_smfrd_paired \
  --eval-data-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_final_runs/smfrd_train_rmfrd_eval_seed42 \
  --train-identities 1000 \
  --eval-identities 80 \
  --max-train-images-per-condition 4 \
  --max-eval-images-per-condition 8 \
  --train-pairs-per-case 6000 \
  --eval-pairs-per-case 800 \
  --epochs 50 \
  --dropout 0.30 \
  --seed 42
```

## Notes

- The LFW-SMFRD Kaggle mirror contains simulated masked LFW images. The pair-head
  training root still needs corresponding unmasked images for the same
  identities.
- If a downloaded root has both masked and unmasked condition folders already,
  pass that root directly as `--train-data-root`; no paired view is needed.
- Keep the RMFRD real split as evaluation data so the extension does not become
  a synthetic-only result.

## Sources

- Official RMFD/RMFRD/SMFRD repository:
  https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
- RMFD summary with dataset sizes:
  https://hyper.ai/en/datasets/19431
- LFW-SMFRD Kaggle mirror:
  https://www.kaggle.com/datasets/muhammeddalkran/lfw-simulated-masked-face-dataset
