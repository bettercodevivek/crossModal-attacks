# Cross-Modal Adversarial Attacks

A comprehensive robustness testing framework for multimodal models (CLIP) implementing cross-modal adversarial attacks as described in "Cross-Modal Adversarial Attacks: Towards Generalized Robustness Testing in Deep Learning".

## Overview

This framework implements multiple adversarial attack methods that perturb images to manipulate text-based predictions in CLIP, demonstrating cross-modal vulnerability in vision-language models.

## Project Structure

```
src/
├── demo_attack.py          # Main script for running attacks
├── config.py              # Configuration parameters
├── utils.py               # Utility functions
│
├── attacks/               # Attack implementations
│   ├── __init__.py
│   ├── patch_attack.py    # Universal adversarial patch
│   ├── fgsm_attack.py     # Fast Gradient Sign Method
│   └── pgd_attack.py      # Projected Gradient Descent
│
├── evaluation/            # Evaluation modules
│   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics (ASR, confidence shift, robustness)
│   └── robustness_evaluator.py  # Robustness evaluation pipeline
│
├── visualization/         # Visualization tools
│   ├── __init__.py
│   └── visualize_results.py  # Generate comparison visualizations
│
└── results/               # Output directory
    ├── images/            # Visualization outputs
    └── metrics.json       # Evaluation metrics
```

## Features

### Attack Methods

1. **Universal Adversarial Patch** (`patch`)
   - Generates a universal patch that can be applied to any image
   - Optimized via gradient descent to maximize similarity with target caption
   - Configurable patch size, training steps, and placement

2. **FGSM Attack** (`fgsm`)
   - Fast Gradient Sign Method
   - Single-step attack: `x_adv = x + epsilon * sign(grad(loss))`
   - Configurable epsilon (perturbation budget)

3. **PGD Attack** (`pgd`)
   - Projected Gradient Descent
   - Iterative multi-step attack with projection
   - Configurable epsilon, steps, and step size (alpha)

### Evaluation Metrics

- **Attack Success Rate (ASR)**: Percentage of successful attacks
- **Confidence Shift**: Average change in similarity to target caption
- **Robustness Score**: `R = 1 - ASR`

### Cross-Modal Objective

All attacks optimize the same objective:
- Maximize cosine similarity between image embedding and target caption embedding
- Demonstrates how image perturbations affect language/text outputs

## Usage

### Basic Usage

Run attacks from the `src/` directory:

```bash
cd src
python demo_attack.py --attack patch
python demo_attack.py --attack fgsm
python demo_attack.py --attack pgd
```

### Advanced Usage

```bash
# Specify custom directories
python demo_attack.py --attack patch --train_dir data/images --eval_dir data/holdout
python demo_attack.py --attack fgsm --eval_dir data/holdout
python demo_attack.py --attack pgd --eval_dir data/holdout
```

### Configuration

Edit `config.py` to adjust parameters:

- **Patch Attack**: `PATCH_SIZE`, `PATCH_STEPS`, `PATCH_LR`, `PATCH_LOCATION`
- **FGSM Attack**: `FGSM_EPSILON`
- **PGD Attack**: `PGD_EPSILON`, `PGD_STEPS`, `PGD_ALPHA`
- **Target Caption**: `TARGET_TEXT`

## Outputs

### Metrics

Results are saved to `results/metrics.json`:

```json
[
  {
    "attack": "PatchAttack",
    "asr": 0.84,
    "avg_confidence_shift": 0.42,
    "robustness_score": 0.16,
    "num_images": 50
  }
]
```

### Visualizations

Comparison images are saved to `results/images/` showing:
- Original image with similarity score
- Adversarial image with similarity score

## Requirements

- Python 3.7+
- PyTorch
- transformers (Hugging Face)
- torchvision
- PIL/Pillow
- matplotlib
- numpy
- tqdm

## Installation

```bash
pip install torch torchvision transformers pillow matplotlib numpy tqdm
```

## Example Workflow

1. **Prepare Data**: Place training images in `data/images/` and evaluation images in `data/holdout/`

2. **Run Patch Attack**:
   ```bash
   python demo_attack.py --attack patch
   ```
   - Trains universal patch on training images
   - Evaluates on holdout images
   - Generates metrics and visualizations

3. **Run FGSM Attack**:
   ```bash
   python demo_attack.py --attack fgsm
   ```
   - Generates FGSM adversarial examples
   - Computes evaluation metrics
   - Saves visualizations

4. **Run PGD Attack**:
   ```bash
   python demo_attack.py --attack pgd
   ```
   - Generates PGD adversarial examples
   - Computes evaluation metrics
   - Saves visualizations

## Research Context

This implementation demonstrates cross-modal adversarial attacks where:
- **Input Modality**: Images (visual)
- **Output Modality**: Text/Language (CLIP text embeddings)
- **Attack Goal**: Perturb images to maximize similarity with target caption

This reveals vulnerabilities in multimodal models where perturbations in one modality (vision) can manipulate outputs in another modality (language).

## License

See LICENSE file for details.
