# Offline Course Demo

Open `index.html` in a browser to run the curated verifier. The checked-in
bundle uses anonymized schematic assets so the demo works without Colab,
internet access, or private dataset images.

For the final presentation dataset bundle, rebuild this folder from a Colab run:

```bash
python scripts/export_demo_bundle.py \
  --scores /path/to/pair_head_robustness_pair_scores.csv \
  --image-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir demo \
  --seed 42 \
  --target-far 0.05
```

The exporter writes `assets/demo_pairs.csv`, `assets/demo_scores.csv`,
`assets/thresholds.json`, `assets/summary.json`, `assets/demo_data.js`, and the
copied per-example image/view assets used by the static app.
