# RMFRD Periocular Specialist Probe

Run date: 2026-04-29

Command:

```bash
python scripts/probe_periocular_specialist.py \
  --data-root /content/datasets/rmfrd/extracted/self-built-masked-face-recognition-dataset \
  --out-dir /content/masked_face_spike_results/periocular_seed42 \
  --train-identities 140 \
  --eval-identities 80 \
  --max-images-per-condition 8 \
  --train-pairs-per-case 600 \
  --eval-pairs-per-case 600 \
  --epochs 30 \
  --steps-per-epoch 80 \
  --identities-per-batch 16 \
  --samples-per-identity 4 \
  --embed-batch-size 128 \
  --alpha-grid-size 21 \
  --seed 42
```

Notes:

- MediaPipe Tasks Face Landmarker produced periocular crops for 2,209 / 2,582 selected images.
- MTCNN/FaceNet full-face alignment succeeded for 2,125 / 2,582 selected images.
- Evaluation pairs in the final metrics are the sampled pairs that had both full-face and periocular embeddings.
- The tuned alpha selected on training identities was 0.80, but it overfit and underperformed on held-out identities.
