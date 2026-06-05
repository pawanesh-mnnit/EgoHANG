# EgoHAnG: Graph-Enhanced Horizon Aware Egocentric Action Anticipation

[![ICPR 2026](https://img.shields.io/badge/ICPR-2026-blue)](https://icpr2026.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-orange)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

> **Pawanesh Kumar Vishwakarma, Ananda S. Chowdhury, Abhimanyu Sahu**
> MNNIT Allahabad · Jadavpur University
> ICPR 2026 — Paper ID 911

---

## Overview

EgoHAnG is an end-to-end framework for egocentric action anticipation that integrates:

1. **Early multimodal fusion** — RGB + Optical Flow via pretrained ResNet-50 + PCA
2. **Graph-Enhanced Temporal Reasoning (GETR)** — k-NN similarity graph + Graph Attention Network (GAT) to capture inter-frame relationships, fused with a Transformer encoder for global temporal context
3. **Horizon-Aware Transformer Decoder (HATD)** — learnable horizon-specific queries to independently predict verb, noun, and action at multiple future anticipation times

![Architecture](Viz_Results/ActionAnticipation1.png)

---

## Results

### EPIC-Kitchens

| Method | Top-1 Verb | Top-1 Action | Top-5 Action | Top-5 Recall |
|--------|-----------|-------------|-------------|-------------|
| RULSTM (ICCV'19) | 33.04 | 14.39 | 33.73 | 13.30 |
| InAViT (WACV'24) | 49.14 | 23.75 | — | 25.89 |
| **EgoHAnG (Ours)** | **50.85** | **24.40** | **35.64** | **28.71** |

### EGTEA Gaze+

| Method | Top-1 Verb | Top-1 Noun | Top-1 Action | Top-5 Recall |
|--------|-----------|-----------|-------------|-------------|
| InAViT (WACV'24) | 79.30 | 77.60 | 67.80 | 58.20 |
| **EgoHAnG (Ours)** | **89.22** | **82.67** | **71.40** | **70.03** |

> **Note:** Pretrained weights are provided per participant. Results above are aggregated across all participants as reported in the paper.

---

## Repository Structure

```
EgoHANG/
├── extract_features.py     # Step 1: Feature extraction
├── train.py                # Step 2: Training
├── evaluate.py             # Step 3: Evaluation
├── model.py                # Model architecture (GETR + HATD)
├── dataset.py              # Dataset loader
├── requirements.txt        # Python dependencies
├── tools/
│   ├── imports.py          # Common imports
│   ├── pca.py              # PCA utility class
│   └── pca_2048_to_512.pkl # Pretrained PCA model
├── EPIC-Kitchens/
│   └── Labels/             # Action label CSVs per participant
├── EGTEA/
│   └── Labels/             # Action label CSVs per participant
├── Train_Val/              # Training/validation split files
└── Viz_Results/            # Architecture diagrams
```

---

## Installation

### Requirements

- Python **3.10** or higher
- CUDA-compatible GPU (tested on NVIDIA RTX A4000 and RTX 4050)
- 16 GB+ RAM

### Step 1 — Clone the repository

```bash
git clone https://github.com/pawanesh-mnnit/EgoHANG.git
cd EgoHANG
```

### Step 2 — Create a virtual environment

```bash
python -m venv egohang_env

# Linux / macOS
source egohang_env/bin/activate

# Windows (Command Prompt)
egohang_env\Scripts\activate

# Windows (Git Bash)
source egohang_env/Scripts/activate
```

### Step 3 — Install PyTorch

Check your CUDA version first:
```bash
nvidia-smi
```

Then install PyTorch matching your CUDA version:
```bash
# CUDA 12.1 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Step 4 — Install PyTorch Geometric

```bash
pip install torch-geometric

# CUDA 12.1
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# CUDA 11.8
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# CPU only
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify installation

```bash
python -c "
import torch
import torchvision
import torch_geometric
import pandas
import numpy
import sklearn
import joblib
print('torch         :', torch.__version__)
print('torchvision   :', torchvision.__version__)
print('torch_geometric:', torch_geometric.__version__)
print('CUDA available :', torch.cuda.is_available())
print('All OK')
"
```

---

## Quick Sanity Check (No Dataset Required)

Run this to verify the model works before touching any dataset:

```bash
python -c "
import torch
from model import AnticipationModel
model = AnticipationModel(
    feat_dim=512,
    num_classes={'verb': 97, 'noun': 300, 'action': 2513},
    k_fut=8
)
dummy = torch.randn(2, 90, 512)
out = model(dummy)
print('verb shape   :', out['verb'].shape)
print('noun shape   :', out['noun'].shape)
print('action shape :', out['action'].shape)
print('Model OK')
"
```

Expected output:
```
verb shape   : torch.Size([2, 8, 97])
noun shape   : torch.Size([2, 8, 300])
action shape : torch.Size([2, 8, 2513])
Model OK
```

---

## Quick Reproduction (Recommended Starting Point)

To quickly verify the code works **without extracting features from scratch**,
download the pretrained weights and pre-extracted feature CSVs directly from
the GitHub Release. No dataset download is required.

### EPIC-Kitchens — P01_04

**Step 1** — Download from GitHub Releases:

| File | Link |
|------|------|
| Pretrained weight | [P01_04_fused_model_PCA.pth](https://github.com/pawanesh-mnnit/EgoHANG/releases/download/v1.0.0/P01_04_fused_model_PCA.pth) |
| Fused features CSV | [P01_04_fused_features_PCA.csv](https://github.com/pawanesh-mnnit/EgoHANG/releases/download/v1.0.0/P01_04_fused_features_PCA.csv) |

Place files in the correct folders:
```bash
mkdir checkpoints
# place P01_04_fused_model_PCA.pth -> checkpoints/
# place P01_04_fused_features_PCA.csv -> EPIC-Kitchens/Features/
```

**Step 2** — Label CSV is already in the repo at:
`EPIC-Kitchens/Labels/P01_04.csv`

**Step 3** — Run evaluation:
```bash
python evaluate.py \
    --dataset epic_kitchens \
    --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \
    --label_csv EPIC-Kitchens/Labels/P01_04.csv \
    --model_path checkpoints/P01_04_fused_model_PCA.pth
```

---

### EGTEA Gaze+ — OP01-R01-PastaSalad

**Step 1** — Download from GitHub Releases:

| File | Link |
|------|------|
| Pretrained weight | [OP01-R01-PastaSalad_Fused_model.pth](https://github.com/pawanesh-mnnit/EgoHANG/releases/download/v1.0.0/OP01-R01-PastaSalad_Fused_model.pth) |
| Fused features CSV | [OP01-R01-PastaSalad_fused.csv](https://github.com/pawanesh-mnnit/EgoHANG/releases/download/v1.0.0/OP01-R01-PastaSalad_fused.csv) |

Place files in the correct folders:
```bash
# place OP01-R01-PastaSalad_Fused_model.pth -> checkpoints/
# place OP01-R01-PastaSalad_fused.csv -> EGTEA/Features/
```

**Step 2** — Label CSV is already in the repo at:
`EGTEA/Labels/OP01-R01.csv`

**Step 3** — Run evaluation:
```bash
python evaluate.py \
    --dataset egtea \
    --fused_csv EGTEA/Features/OP01-R01-PastaSalad_fused.csv \
    --label_csv EGTEA/Labels/OP01-R01.csv \
    --model_path checkpoints/OP01-R01-PastaSalad_Fused_model.pth
```

> **Note:** All pretrained weights and fused feature CSVs are available
> directly from the [GitHub Releases v1.0.0](https://github.com/pawanesh-mnnit/EgoHANG/releases/tag/v1.0.0)
> page. No dataset download is required to run the Quick Reproduction steps.

---

## Datasets

### EPIC-Kitchens
Download from the official website: https://epic-kitchens.github.io/

Expected folder structure:
```
/path/to/EPIC_Kitchens/
├── RGB/
│   ├── P01_04/   (frames as .jpg)
│   ├── P01_05/
│   └── ...
└── OpticalFlow/
    ├── P01_04/
    ├── P01_05/
    └── ...
```

### EGTEA Gaze+
Download from: https://cbs.ic.gatech.edu/fpv/

Expected folder structure:
```
/path/to/EGTEA/
├── RGB/
│   ├── OP01-R01-PastaSalad/
│   └── ...
└── Flow/
    ├── OP01-R01-PastaSalad/
    └── ...
```

---

## Pretrained Models

Download all pretrained model weights and feature CSVs from the GitHub Release:

**[Download from GitHub Releases v1.0.0](https://github.com/pawanesh-mnnit/EgoHANG/releases/tag/v1.0.0)**

| File | Type | Dataset | Participant |
|------|------|---------|------------|
| `P01_04_fused_model_PCA.pth` | Model | EPIC-Kitchens | P01_04 |
| `P01_04_Fused_model.pth` | Model | EPIC-Kitchens | P01_04 |
| `P01_05_Fused_model.pth` | Model | EPIC-Kitchens | P01_05 |
| `P01_04_fused_features_PCA.csv` | Features | EPIC-Kitchens | P01_04 |
| `P01_05_fused_features.csv` | Features | EPIC-Kitchens | P01_05 |
| `OP01-R01-PastaSalad_Fused_model.pth` | Model | EGTEA Gaze+ | OP01-R01 |
| `OP01-R02-TurkeySandwich_Fused_model.pth` | Model | EGTEA Gaze+ | OP01-R02 |
| `OP01-R03-BaconAndEggs_Fused_model.pth` | Model | EGTEA Gaze+ | OP01-R03 |
| `OP01-R04-ContinentalBreakfast_Fused_model.pth` | Model | EGTEA Gaze+ | OP01-R04 |
| `OP01-R05-Cheeseburger_Fused_model.pth` | Model | EGTEA Gaze+ | OP01-R05 |
| `OP01-R01-PastaSalad_fused.csv` | Features | EGTEA Gaze+ | OP01-R01 |
| `OP01-R02-TurkeySandwich_fused.csv` | Features | EGTEA Gaze+ | OP01-R02 |
| `OP01-R03-BaconAndEggs_fused.csv` | Features | EGTEA Gaze+ | OP01-R03 |
| `OP01-R04-ContinentalBreakfast_fused.csv` | Features | EGTEA Gaze+ | OP01-R04 |
| `OP01-R05-Cheeseburger_fused.csv` | Features | EGTEA Gaze+ | OP01-R05 |

> **Note:** Each `.pth` file contains the model trained on a single participant.
> Class counts are automatically inferred from the checkpoint — no manual
> configuration needed.

---

## Usage

### Step 1 — Extract Features

```bash
# EPIC-Kitchens
python extract_features.py \
    --dataset epic_kitchens \
    --rgb_root /path/to/EPIC_Kitchens/RGB \
    --flow_root /path/to/EPIC_Kitchens/OpticalFlow \
    --labels_root EPIC-Kitchens/Labels \
    --output_root EPIC-Kitchens/Features \
    --pca_path tools/pca_2048_to_512.pkl

# EGTEA Gaze+
python extract_features.py \
    --dataset egtea \
    --rgb_root /path/to/EGTEA/RGB \
    --flow_root /path/to/EGTEA/Flow \
    --labels_root EGTEA/Labels \
    --output_root EGTEA/Features \
    --pca_path tools/pca_2048_to_512.pkl
```

### Step 2 — Train

```bash
# EPIC-Kitchens
python train.py \
    --dataset epic_kitchens \
    --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \
    --label_csv EPIC-Kitchens/Labels/P01_04.csv \
    --save_path checkpoints/P01_04_model.pth

# EGTEA Gaze+
python train.py \
    --dataset egtea \
    --fused_csv EGTEA/Features/OP01-R01-PastaSalad_fused.csv \
    --label_csv EGTEA/Labels/OP01-R01.csv \
    --save_path checkpoints/OP01-R01_model.pth
```

### Step 3 — Evaluate with Pretrained Weights

```bash
# EPIC-Kitchens
python evaluate.py \
    --dataset epic_kitchens \
    --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \
    --label_csv EPIC-Kitchens/Labels/P01_04.csv \
    --model_path checkpoints/P01_04_fused_model_PCA.pth

# EGTEA Gaze+
python evaluate.py \
    --dataset egtea \
    --fused_csv EGTEA/Features/OP01-R01-PastaSalad_fused.csv \
    --label_csv EGTEA/Labels/OP01-R01.csv \
    --model_path checkpoints/OP01-R01-PastaSalad_Fused_model.pth
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Observation Window (T_obs) | 90 frames |
| Anticipation Horizons | 2.0, 1.75, 1.50, 1.25, 1.0, 0.75, 0.50, 0.25 s |
| Feature Dimension | 512 (after PCA) |
| k-NN Graph Neighbours | 5 |
| GAT Layers | 3 |
| GAT Heads | 8 |
| Transformer Layers | 3 |
| Batch Size | 8 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Epochs | 100 |
| Dropout | 0.1 |
| RGB Fusion Weight (α) | 0.6 |
| Flow Fusion Weight (1-α) | 0.4 |
| EPIC-Kitchens FPS | 60.0 |
| EGTEA Gaze+ FPS | 24.0 |

---

## Train/Validation Split

Each model is trained and evaluated on a single participant.
Training uses 80% of the participant's annotated actions and
validation uses the remaining 20%, split randomly inside `train.py`
(controlled by `VAL_SPLIT = 0.2`).


## Model Efficiency

| Method | GFLOPs | Parameters | Latency |
|--------|--------|-----------|---------|
| InAViT (WACV'24) | 391 | 157.2M | — |
| EgoHAnG w/o PCA | 600 | 398.1M | 21.58 ms |
| **EgoHAnG (Ours)** | **150** | **26.3M** | **19.34 ms** |

*Latency measured on NVIDIA RTX A4000 GPU, averaged over 100 runs.*

---

## Tested Environment

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torch-geometric | 2.7.0 |
| numpy | 1.24+ |
| pandas | 2.0+ |
| scikit-learn | 1.3+ |
| GPU | NVIDIA RTX 4050 / RTX A4000 |
| CUDA | 12.1 |

---

## License

This project is released under the [CC BY-NC 4.0 License](LICENSE).
Free for academic and research use. Commercial use requires permission.

---

## Contact

- Pawanesh Kumar Vishwakarma — pawanesh.2023rcs04@mnnit.ac.in
