# Cross-Modal Adversarial Attacks

A comprehensive robustness testing framework for the paper *Cross-Modal Adversarial Attacks: Towards Generalized Robustness Testing in Deep Learning*: **CLIP** cross-modal attacks (vision → language) plus **paper Section V–VI** baselines (MNIST/CIFAR-10 CNNs, FGSM/PGD/CW, ASR, adversarial training, distillation).

## Overview

**CLIP path** (`demo_attack.py`): adversarial methods that perturb images to change text-side behavior (similarity to a target caption), illustrating shared-embedding vulnerability.

**Paper CNN path** (`src/paper/`): the MNIST and CIFAR-10 experiments with classification-focused FGSM, PGD, optional Carlini–Wagner–style L2 attack, robustness metrics, and defenses—matching the empirical methodology described in Sections V–VI.

## Project Structure

```
src/
├── demo_attack.py          # CLIP cross-modal attacks (main script)
├── config.py              # CLIP / patch / PGD config
├── utils.py               # Utility functions
│
├── paper/                 # Paper Sec. V–VI: MNIST/CIFAR CNN, FGSM/PGD/CW, defenses
│   ├── run_paper_experiments.py
│   ├── models.py
│   ├── data.py
│   ├── classification_attacks.py
│   ├── cw_attack.py
│   ├── train_models.py
│   ├── metrics_classify.py
│   ├── plots.py
│   └── paper_config.py
│
├── attacks/               # CLIP attack implementations
│   ├── __init__.py
│   ├── patch_attack.py    # Universal adversarial patch
│   ├── fgsm_attack.py     # Fast Gradient Sign Method
│   └── pgd_attack.py      # Projected Gradient Descent
│
├── evaluation/            # CLIP evaluation modules
│   ├── __init__.py
│   ├── metrics.py         # Similarity-based ASR, confidence shift, robustness
│   └── robustness_evaluator.py
│
├── visualization/
│   ├── __init__.py
│   └── visualize_results.py
│
└── results/               # CLIP run outputs
    ├── images/
    └── metrics.json
```

At the repo root, `paper_checkpoints/` and `paper_results/` are created by the paper CLI.

## Paper-aligned experiments (MNIST / CIFAR-10 CNNs)

Section VI of *Cross-Modal Adversarial Attacks: Towards Generalized Robustness Testing in Deep Learning* reports **white-box FGSM/PGD** on **CNN classifiers** trained on **MNIST** and **CIFAR-10**, with **attack success rate (ASR)** and **robustness** \(R = 1 - \mathrm{ASR}\). That pipeline is implemented under `src/paper/`:

| Component | Location |
|-----------|----------|
| MNIST CNN / deeper CIFAR-10 CNN | `src/paper/models.py` |
| FGSM, PGD (\(L_\infty\)), CW-style L2 | `src/paper/classification_attacks.py`, `src/paper/cw_attack.py` |
| Metrics (ASR, \(R\)) | `src/paper/metrics_classify.py` |
| Standard training, adversarial training, defensive distillation | `src/paper/train_models.py` |
| CLI | `src/paper/run_paper_experiments.py` |

**Train** a classifier (optional defenses: `none`, `adversarial`, `distillation`):

```bash
# From project root
python src/paper/run_paper_experiments.py train --dataset mnist --defense none
python src/paper/run_paper_experiments.py train --dataset cifar10 --defense adversarial
python src/paper/run_paper_experiments.py train --dataset mnist --defense distillation --teacher_ckpt paper_checkpoints/mnist_none.pt
```

**Evaluate** FGSM, PGD, or CW on a saved checkpoint:

```bash
python src/paper/run_paper_experiments.py eval --dataset mnist --checkpoint paper_checkpoints/mnist_none.pt --attack fgsm
python src/paper/run_paper_experiments.py eval --dataset mnist --checkpoint paper_checkpoints/mnist_none.pt --attack pgd
python src/paper/run_paper_experiments.py eval --dataset mnist --checkpoint paper_checkpoints/mnist_none.pt --attack cw --limit_batches 10
```

**Reproduce Table 2–style ASR** (trains MNIST + CIFAR unless checkpoints exist; use `--quick` for a short run):

```bash
python src/paper/run_paper_experiments.py reproduce-table --defense none
python src/paper/run_paper_experiments.py reproduce-table --quick --limit_batches 20
```

**Fast CNN benchmark (~2–3 minutes, git-friendly JSON + plots):**

```bash
python src/paper/run_paper_experiments.py fast-benchmark
```

Writes to **`cnn_paper_benchmark/`** (committed): `table2_asr.json`, `asr_bars.png`, `asr_radar_fgsm_pgd.png`. Model checkpoints still go to **`paper_checkpoints/`** (ignored). Use `--skip_train` if you already have `mnist_none.pt` and `cifar10_none.pt`.

Outputs: `paper_results/table2_asr.json`, `paper_results/asr_bars.png`, `paper_results/asr_radar_fgsm_pgd.png`. Checkpoints go to `paper_checkpoints/`. Torchvision downloads datasets under `data/` (project root).

The **CLIP cross-modal** demos (`demo_attack.py`) remain the multimodal (vision → language) baseline described in the paper’s introduction; the **`paper/`** module matches the **CNN + MNIST/CIFAR** experimental setup in Sections V–VI.

---

## Setup & Running Guide

Everything you need to go from a fresh clone to running all experiments.

### Prerequisites

- Python 3.8+
- Git
- GPU with CUDA (optional but recommended — CPU works for all experiments)

---

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd crossModal-attacks
```

---

### Step 2 — Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies

**CPU only (works everywhere):**
```bash
pip install torch torchvision transformers pillow matplotlib numpy tqdm requests
```

**With CUDA GPU (faster training and attacks):**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install transformers pillow matplotlib numpy tqdm requests
```

**Verify the install:**
```bash
python -c "import torch; import transformers; print('Ready:', torch.__version__)"
```

---

### Step 4 — Download demo images (for CLIP track only)

The CLIP cross-modal attacks need real images — the included script downloads 30 training images and 10 holdout images automatically:

```bash
python download_images.py
```

This creates:
```
data/
├── images/     ← 30 images used to train the universal patch
└── holdout/    ← 10 images used to evaluate all CLIP attacks
```

Images are sourced from [Lorem Picsum](https://picsum.photos/) (free, no API key). Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`.

> The CNN experiments (Track 2) download MNIST and CIFAR-10 automatically via torchvision — no manual download needed.

---

### Step 5 — Run the experiments

There are two independent tracks. Run either or both.

---

#### Track 1: CLIP Cross-Modal Attacks

All commands run from the `src/` directory. Each attack perturbs images to increase their CLIP cosine similarity to a target caption (`"a photo of a banana"` by default).

```bash
cd src
```

**Universal Adversarial Patch**
```bash
python demo_attack.py --attack patch
```
- Trains a single 100×100 pixel patch over 800 gradient steps using Adam
- The patch is "universal" — once trained on `data/images/`, it transfers to unseen images
- Applies the trained patch to all images in `data/holdout/`
- **Output:** `src/output/universal_patch.png` (the patch itself), side-by-side comparison PNGs in `src/results/images/`, metrics appended to `src/results/metrics.json`
- **Runtime:** 5–15 minutes depending on hardware

**FGSM Attack**
```bash
python demo_attack.py --attack fgsm
```
- Single-step attack: computes the gradient of CLIP similarity w.r.t. the image, then steps in the sign direction by ε = 0.03
- Fast but weaker than PGD
- **Output:** comparison PNGs in `src/results/images/`, metrics appended to `src/results/metrics.json`
- **Runtime:** 1–3 minutes

**PGD Attack**
```bash
python demo_attack.py --attack pgd
```
- Iterative version of FGSM: takes 40 small steps (α = 0.01), projecting back into the L∞ ball (ε = 0.03) after each step
- Stronger attack — finds adversarial examples that FGSM misses
- **Output:** comparison PNGs in `src/results/images/`, metrics appended to `src/results/metrics.json`
- **Runtime:** 3–8 minutes

**Run all three in sequence:**
```bash
python demo_attack.py --attack patch
python demo_attack.py --attack fgsm
python demo_attack.py --attack pgd
```

**Custom image directories:**
```bash
python demo_attack.py --attack patch --train_dir ../data/images --eval_dir ../data/holdout
```

**Change the target caption:** edit `TARGET_TEXT` in `src/config.py`.

---

#### Track 2: CNN Paper Experiments (MNIST / CIFAR-10)

All commands run from the **project root**. Datasets are downloaded automatically on first run into `data/`.

**Step A — Train a model**

```bash
# Standard training (no defense)
python src/paper/run_paper_experiments.py train --dataset mnist --defense none
python src/paper/run_paper_experiments.py train --dataset cifar10 --defense none
```
- Trains a CNN on MNIST (12 epochs) or CIFAR-10 (25 epochs) using Adam
- Saves checkpoint to `paper_checkpoints/<dataset>_<defense>.pt`
- Prints training loss and validation accuracy each epoch

```bash
# Adversarial training defense
python src/paper/run_paper_experiments.py train --dataset mnist --defense adversarial
python src/paper/run_paper_experiments.py train --dataset cifar10 --defense adversarial
```
- Same as above, but each training batch is augmented with PGD adversarial examples (7 inner steps, ε = 0.03)
- Results in a model that is more robust to white-box attacks at the cost of slightly lower clean accuracy

```bash
# Defensive distillation (requires a pre-trained teacher checkpoint)
python src/paper/run_paper_experiments.py train --dataset mnist --defense distillation \
    --teacher_ckpt paper_checkpoints/mnist_none.pt
```
- Trains a student CNN to match the soft probability outputs of the teacher (temperature T = 5.0)
- Smooths the decision boundary, making gradient-based attacks less effective

**Step B — Evaluate attacks on a trained checkpoint**

```bash
# FGSM attack
python src/paper/run_paper_experiments.py eval \
    --dataset mnist \
    --checkpoint paper_checkpoints/mnist_none.pt \
    --attack fgsm
```
- Runs FGSM (ε = 0.03) on the full MNIST test set against the saved checkpoint
- Prints clean accuracy, attack success rate (ASR), and robustness score R = 1 − ASR

```bash
# PGD attack
python src/paper/run_paper_experiments.py eval \
    --dataset cifar10 \
    --checkpoint paper_checkpoints/cifar10_none.pt \
    --attack pgd
```
- Runs PGD (ε = 0.03, 40 steps, α = 0.01) on the full CIFAR-10 test set
- More thorough than FGSM — use this for the paper's primary robustness metric

```bash
# Carlini–Wagner L2 attack (slower — use --limit_batches to cap evaluation)
python src/paper/run_paper_experiments.py eval \
    --dataset mnist \
    --checkpoint paper_checkpoints/mnist_none.pt \
    --attack cw \
    --limit_batches 10
```
- Optimization-based attack that minimizes L2 distortion while causing misclassification
- Slower per-image than FGSM/PGD; `--limit_batches` limits how many test batches are evaluated

**Step C — Reproduce the results table**

```bash
# Full run (trains both MNIST and CIFAR-10 if checkpoints don't exist, then evaluates)
python src/paper/run_paper_experiments.py reproduce-table --defense none

# Quick version (fewer batches, useful for testing the pipeline)
python src/paper/run_paper_experiments.py reproduce-table --quick --limit_batches 20
```
- Trains both datasets sequentially (or reuses existing checkpoints)
- Evaluates FGSM and PGD on both datasets
- Saves `paper_results/table2_asr.json`, `paper_results/asr_bars.png`, `paper_results/asr_radar_fgsm_pgd.png`

**Step D — Fast benchmark (2–3 minutes, results committed to repo)**

```bash
python src/paper/run_paper_experiments.py fast-benchmark
```
- Trains on small subsets (8,000 MNIST / 5,000 CIFAR-10) for 1–2 epochs
- Evaluates on 1,024 samples (4 batches × 256) with 5-step PGD
- Writes directly to `cnn_paper_benchmark/` (tracked by git): `table2_asr.json`, `asr_bars.png`, `asr_radar_fgsm_pgd.png`
- Use `--skip_train` if checkpoints already exist:
  ```bash
  python src/paper/run_paper_experiments.py fast-benchmark --skip_train
  ```

---

### Output Files Reference

| Path | Contents | Created by |
|------|----------|------------|
| `src/results/metrics.json` | CLIP attack ASR, confidence shift, robustness per run | `demo_attack.py` |
| `src/results/images/` | Side-by-side original vs. adversarial PNGs | `demo_attack.py` |
| `src/output/universal_patch.png` | Trained universal adversarial patch image | `demo_attack.py --attack patch` |
| `paper_checkpoints/<name>.pt` | Saved CNN model weights | `run_paper_experiments.py train` |
| `paper_results/table2_asr.json` | Full-run ASR table | `reproduce-table` |
| `paper_results/asr_bars.png` | Bar chart of ASR by dataset/attack | `reproduce-table` |
| `paper_results/asr_radar_fgsm_pgd.png` | Radar chart comparing FGSM vs PGD | `reproduce-table` |
| `cnn_paper_benchmark/table2_asr.json` | Fast benchmark ASR table (committed) | `fast-benchmark` |
| `cnn_paper_benchmark/asr_bars.png` | Fast benchmark bar chart (committed) | `fast-benchmark` |

---

### Configuration

| File | Controls |
|------|---------|
| `src/config.py` | CLIP attacks: `PATCH_SIZE`, `PATCH_STEPS`, `PATCH_LR`, `FGSM_EPSILON`, `PGD_EPSILON`, `PGD_STEPS`, `PGD_ALPHA`, `TARGET_TEXT` |
| `src/paper/paper_config.py` | CNN training: epochs, batch size, LR; attack ε/α/steps; CW constants; distillation temperature |

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `No images found` in CLIP track | Run `python download_images.py` from project root first, or pass `--eval_dir ../data/holdout` |
| CUDA out of memory | Reduce `PATCH_BATCH_SIZE` in `config.py` or add `--limit_batches` |
| Slow patch training | Normal on CPU — ~15 min. Add a GPU or reduce `PATCH_STEPS` in `config.py` |
| `ModuleNotFoundError` | Activate the venv and re-run `pip install ...` |
| Checkpoint not found | Run the `train` subcommand first before `eval` |

---

## Features

### Attack Methods

| Attack | Type | Target | Strength |
|--------|------|--------|----------|
| **Universal Patch** | Optimization (Adam) | CLIP similarity | High (unbounded patch size) |
| **FGSM** | Single-step gradient | CLIP similarity / classification | Medium |
| **PGD** | Iterative gradient | CLIP similarity / classification | High |
| **Carlini–Wagner L2** | Optimization (Adam) | Classification (CNN only) | Very high |

### Evaluation Metrics

- **Attack Success Rate (ASR)**: Fraction of successful attacks
  - CLIP: fraction where adversarial similarity > original similarity
  - CNN: fraction of samples misclassified after attack
- **Confidence Shift**: Mean change in CLIP similarity score (CLIP track)
- **Robustness Score**: `R = 1 - ASR` (higher is better)

### Defenses

| Defense | How it works |
|---------|-------------|
| **None** | Standard cross-entropy training |
| **Adversarial training** | Augments each training batch with PGD examples (inner 7 steps) |
| **Defensive distillation** | Student learns soft targets from teacher at temperature T = 5.0 |

## Research Context

This implementation demonstrates cross-modal adversarial attacks where:
- **Input Modality**: Images (visual)
- **Output Modality**: Text/Language (CLIP text embeddings)
- **Attack Goal**: Perturb images to maximize similarity with target caption

This reveals vulnerabilities in multimodal models where perturbations in one modality (vision) can manipulate outputs in another modality (language).

## License

See LICENSE file for details.
