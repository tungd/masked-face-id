## Project Overview

This project implements a phase-based evaluation pipeline to test whether removing mask-occluded facial regions before embedding extraction improves face recognition performance on masked faces.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mask Recognition Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Baseline                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │ Face Image  │→ │ FaceNet     │→ │ Embedding (512-dim) │     │
│  └─────────────┘  └─────────────┘  └─────────────────────┘     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Mask Exclusion                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Face Image  │→ │ MediaPipe   │→ │ Landmarks   │             │
│  └─────────────┘  └─────────────┘  └──────┬──────┘             │
│                                    ┌──────▼──────┐             │
│                                    │ Mask Region │             │
│                                    │ Exclusion   │             │
│                                    └──────┬──────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────▼──────┐             │
│  │ Embedding   │← │ FaceNet     │← │ Masked Img  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Comparison                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Baseline vs Mask Exclusion: Accuracy, ROC-AUC, FAR/FRR │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | Apple MLX |
| Base Model | FaceNet (InceptionResNetV1) |
| Face Detection | MediaPipe Face Mesh (468 landmarks) |
| Dataset | MaskedFace-Net (137K images) |
| Notebooks | Jupyter + ipywidgets |
| Visualization | Matplotlib, Seaborn |

---

## Implementation Status

### ✅ Completed

| Component | Status | Files |
|-----------|--------|-------|
| **Project Setup** | ✅ Complete | `requirements.txt`, `.gitignore`, package structure |
| **Data Module** | ✅ Complete | `src/data/download.py`, `src/data/dataset.py` |
| **Model Module** | ✅ Complete | `src/models/facenet.py` |
| **Landmarks Module** | ✅ Complete | `src/landmarks/detector.py` |
| **Evaluation Module** | ✅ Complete | `src/evaluation/metrics.py` |
| **Notebooks** | ✅ Complete | 5 notebooks (01-05) |
| **Unit Tests** | ✅ Complete | Tests for all modules |

### 📊 Progress Summary

```
Core Implementation
├── Data Pipeline      [██████████] 100%
├── FaceNet Model      [██████████] 100%
├── Landmark Detector  [██████████] 100%
├── Evaluation Metrics [██████████] 100%
└── Jupyter Notebooks  [██████████] 100%

Testing & Validation
├── Unit Tests         [██████████] 100%
├── Integration Tests  [          ]   0%
└── Results Generation [██        ]  20%

Documentation
├── Design Document    [██████████] 100%
├── Implementation Plan[██████████] 100%
└── Results Summary    [          ]   0%
```

### 📁 Project Structure

```
masked-face-id/
├── notebooks/
│   ├── 01_data_exploration.ipynb    ✅ Data download & exploration
│   ├── 02_baseline_facenet.ipynb    ✅ Baseline FaceNet evaluation
│   ├── 03_mask_exclusion.ipynb      ✅ Mask exclusion implementation
│   ├── 04_analysis.ipynb            ✅ Results comparison & analysis
│   └── 05_landmark_demo.ipynb       ✅ Landmark detection demo
├── src/
│   ├── data/
│   │   ├── download.py              ✅ MaskedFace-Net downloader
│   │   └── dataset.py               ✅ Dataset loader & pair generator
│   ├── models/
│   │   └── facenet.py               ✅ FaceNet for MLX
│   ├── landmarks/
│   │   └── detector.py              ✅ MediaPipe landmark detection
│   └── evaluation/
│       └── metrics.py               ✅ Accuracy, ROC-AUC, FAR/FRR
├── tests/
│   ├── test_download.py             ✅
│   ├── test_dataset.py              ✅
│   ├── test_facenet.py              ✅
│   ├── test_landmarks.py            ✅
│   └── test_metrics.py              ✅
├── docs/
│   └── superpowers/
│       ├── specs/                   ✅ Design document
│       └── plans/                   ✅ Implementation plan
├── results/                         📁 Output directory
├── data/                            📁 Data directory
└── models/                          📁 Pretrained weights
```

### 📋 Module Details

| Module | LOC | Description |
|--------|-----|-------------|
| `src/data/download.py` | 93 | MaskedFace-Net download utilities |
| `src/data/dataset.py` | 154 | Dataset loading, image pair generation |
| `src/models/facenet.py` | 135 | FaceNet InceptionResNetV1 for MLX |
| `src/landmarks/detector.py` | 136 | MediaPipe Face Mesh integration |
| `src/evaluation/metrics.py` | 142 | Verification metrics (accuracy, ROC-AUC, FAR@FRR) |

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import mlx; import mediapipe; print('Dependencies OK')"
```

---

## Usage

### Run Full Pipeline

```bash
# Start Jupyter and run notebooks in order
jupyter notebook notebooks/
```

### Individual Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Download MaskedFace-Net, explore dataset statistics |
| `02_baseline_facenet.ipynb` | Evaluate pretrained FaceNet on masked faces |
| `03_mask_exclusion.ipynb` | Implement and evaluate mask region exclusion |
| `04_analysis.ipynb` | Compare results, generate final report |
| `05_landmark_demo.ipynb` | Interactive landmark detection demo |

### Run Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## Datasets

### Primary: MaskedFace-Net

- **Size:** 137,783 images (67,049 correctly masked + 66,734 incorrectly masked)
- **Resolution:** 1024×1024
- **Source:** [GitHub](https://github.com/cabani/MaskedFace-Net)
- **License:** CC BY-NC-SA 4.0

### Download Instructions

```python
from src.data.download import MaskedFaceNetDownloader

downloader = MaskedFaceNetDownloader()
print(downloader.get_download_instructions())
```

Or manually download from Google Drive:
- **CMFD** (Correctly Masked): [Part 1](https://drive.google.com/file/d/17-FCstm8Fz3bDzFgTmOWHa_c39lTR_1P/view), [Part 2](https://drive.google.com/file/d/1XClQlP9_V6UmmnwTyzjF28vlrVHNSw2H/view)
- **IMFD** (Incorrectly Masked): [Part 1](https://drive.google.com/file/d/1gjltyD_MnNWcnd56NnjUOizdi39CUEPF/view), [Part 2](https://drive.google.com/file/d/1qvbcuTHSLBTxQd3wXNAUIYVXBBJCa2WF/view)

---

## Results

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Verification accuracy at optimal threshold |
| **ROC-AUC** | Area under ROC curve |
| **FAR@1%FRR** | False Accept Rate at 1% False Reject Rate |

### Results Table

| Method | Accuracy | ROC-AUC | FAR@1%FRR |
|--------|----------|---------|-----------|
| Baseline (FaceNet) | *pending* | *pending* | *pending* |
| Mask Exclusion | *pending* | *pending* | *pending* |
| **Improvement** | *pending* | *pending* | *pending* |

> Run notebooks to generate actual results. Results will be saved to `results/` directory.

---

## Development Timeline

| Phase | Tasks | Status |
|-------|-------|--------|
| **Setup** | Environment, dependencies, project structure | ✅ Complete |
| **Data** | Download utilities, dataset loader | ✅ Complete |
| **Model** | FaceNet implementation for MLX | ✅ Complete |
| **Landmarks** | MediaPipe integration, mask inference | ✅ Complete |
| **Evaluation** | Metrics, visualization utilities | ✅ Complete |
| **Notebooks** | Pipeline notebooks (01-05) | ✅ Complete |
| **Testing** | Unit tests for all modules | ✅ Complete |
| **Execution** | Run pipeline, generate results | ⏳ Pending |
| **Analysis** | Final report, conclusions | ⏳ Pending |

---

## References

1. **FaceNet:** Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
2. **MaskedFace-Net:** Cabani et al., "MaskedFace-Net - A dataset of correctly/incorrectly masked face images" (2020)
3. **MediaPipe Face Mesh:** https://developers.google.com/mediapipe/solutions/vision/face_mesh
4. **Apple MLX:** https://ml-explore.github.io/mlx/

---

## License

CC BY-NC-SA 4.0 (for MaskedFace-Net dataset). Code is provided for educational purposes.
