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

### Quick Start

After completing the setup steps above, you can run attacks from the `src/` directory:

```bash
cd src
python demo_attack.py --attack patch    # Universal patch attack
python demo_attack.py --attack fgsm    # FGSM attack
python demo_attack.py --attack pgd     # PGD attack
```

For detailed step-by-step testing instructions, see the [Testing All Attacks](#testing-all-attacks) section above.

### Command-Line Options

```bash
python demo_attack.py --attack <attack_type> [OPTIONS]

Required:
  --attack {patch,fgsm,pgd}    Attack method to use

Optional:
  --train_dir PATH              Training image directory (for patch attack)
                                Default: data/images
  --eval_dir PATH               Evaluation image directory
                                Default: data/holdout
```

### Configuration

Edit `src/config.py` to adjust attack parameters:

- **Patch Attack**: `PATCH_SIZE`, `PATCH_STEPS`, `PATCH_LR`, `PATCH_LOCATION`
- **FGSM Attack**: `FGSM_EPSILON` (perturbation budget)
- **PGD Attack**: `PGD_EPSILON`, `PGD_STEPS`, `PGD_ALPHA`
- **Target Caption**: `TARGET_TEXT` (default: "a photo of a banana")

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

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git (for cloning the repository)
- CUDA-capable GPU (optional, but recommended for faster execution)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd crossModal-attacks
```

Replace `<repository-url>` with the actual repository URL (e.g., `https://github.com/username/crossModal-attacks.git`).

### Step 2: Set Up Python Environment

It's recommended to use a virtual environment to avoid dependency conflicts:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages:

```bash
pip install torch torchvision transformers pillow matplotlib numpy tqdm
```

**Note:** If you have a CUDA-capable GPU and want to use it, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install transformers pillow matplotlib numpy tqdm
```

### Step 4: Prepare Data Directories

The framework expects the following directory structure:

```
crossModal-attacks/
├── data/
│   ├── images/          # Training images (for patch attack)
│   └── holdout/         # Evaluation/holdout images (for all attacks)
├── src/
│   ├── demo_attack.py
│   ├── config.py
│   └── ...
└── ...
```

**Important:** 
- Place your training images in `data/images/` directory (relative to project root)
- Place your evaluation images in `data/holdout/` directory (relative to project root)
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- The scripts are run from the `src/` directory, but paths in `config.py` are relative to the project root

The repository already includes sample images in these directories. You can use them for testing or replace them with your own images.

**Note:** If you need to use custom data directories, you can specify them via command-line arguments (see [Usage](#usage) section).

### Step 5: Verify Installation

Navigate to the `src` directory and verify the setup:

```bash
cd src
python -c "import torch; import transformers; print('Installation successful!')"
```

If you see "Installation successful!", you're ready to proceed.

## Testing All Attacks

This section provides complete step-by-step instructions to test all three attack methods.

### Overview

The framework implements three attack methods:
1. **Patch Attack** - Universal adversarial patch (requires training images)
2. **FGSM Attack** - Fast Gradient Sign Method
3. **PGD Attack** - Projected Gradient Descent

All attacks are run from the `src/` directory. Results are saved to `src/results/` and `src/output/`.

### Test 1: Universal Adversarial Patch Attack

The patch attack generates a universal patch that can be applied to any image to manipulate CLIP's predictions.

**Steps:**

1. Navigate to the `src` directory (all commands should be run from here):
   ```bash
   cd src
   ```
   
   **Important:** All commands should be run from the `src/` directory. The default paths in `config.py` (`data/images` and `data/holdout`) are relative to the project root. When running from `src/`, you may need to either:
   - Use command-line arguments with relative paths: `--train_dir ../data/images --eval_dir ../data/holdout`
   - Or update `config.py` to use `../data/images` and `../data/holdout`

2. Run the patch attack:
   ```bash
   python demo_attack.py --attack patch
   ```
   
   If you get an error about missing image directories, use explicit paths:
   ```bash
   python demo_attack.py --attack patch --train_dir ../data/images --eval_dir ../data/holdout
   ```

3. **What happens:**
   - Loads CLIP model (this may take a minute on first run)
   - Loads training images from `data/images/` (default: 30 images)
   - Generates a universal patch optimized over 800 steps (this takes several minutes)
   - Saves the patch to `output/universal_patch.png`
   - Applies the patch to evaluation images from `data/holdout/`
   - Computes attack success rate and other metrics
   - Generates visualization comparisons
   - Saves results to `results/metrics.json`

4. **Expected output:**
   - Console output showing progress and final metrics
   - Patch image saved to `src/output/universal_patch.png`
   - Visualization images saved to `src/results/images/`
   - Metrics appended to `src/results/metrics.json`

5. **Expected duration:** 5-15 minutes (depending on GPU availability)

**Using custom directories:**
```bash
# Paths are relative to the src/ directory
python demo_attack.py --attack patch --train_dir ../data/images --eval_dir ../data/holdout
```

### Test 2: FGSM Attack

The FGSM (Fast Gradient Sign Method) attack is a single-step attack that perturbs images based on the gradient sign.

**Steps:**

1. Navigate to the `src` directory (if not already there):
   ```bash
   cd src
   ```

2. Run the FGSM attack:
   ```bash
   python demo_attack.py --attack fgsm
   ```
   
   If you get an error about missing image directories, use explicit paths:
   ```bash
   python demo_attack.py --attack fgsm --eval_dir ../data/holdout
   ```

3. **What happens:**
   - Loads CLIP model (if not already loaded)
   - Loads evaluation images from `data/holdout/` (default: 10 images)
   - Generates adversarial perturbations using FGSM with epsilon=0.03
   - Computes attack success rate and confidence shifts
   - Generates visualization comparisons
   - Saves results to `results/metrics.json`

4. **Expected output:**
   - Console output showing progress and final metrics
   - Visualization images saved to `src/results/images/`
   - Metrics appended to `src/results/metrics.json`

5. **Expected duration:** 1-3 minutes

**Using custom directory:**
```bash
python demo_attack.py --attack fgsm --eval_dir ../data/holdout
```

### Test 3: PGD Attack

The PGD (Projected Gradient Descent) attack is an iterative multi-step attack that provides stronger adversarial examples.

**Steps:**

1. Navigate to the `src` directory (if not already there):
   ```bash
   cd src
   ```

2. Run the PGD attack:
   ```bash
   python demo_attack.py --attack pgd
   ```
   
   If you get an error about missing image directories, use explicit paths:
   ```bash
   python demo_attack.py --attack pgd --eval_dir ../data/holdout
   ```

3. **What happens:**
   - Loads CLIP model (if not already loaded)
   - Loads evaluation images from `data/holdout/` (default: 10 images)
   - Generates adversarial perturbations using PGD with:
     - Epsilon: 0.03
     - Steps: 40 iterations
     - Alpha: 0.01 (step size)
   - Computes attack success rate and confidence shifts
   - Generates visualization comparisons
   - Saves results to `src/results/metrics.json`

4. **Expected output:**
   - Console output showing progress and final metrics
   - Visualization images saved to `src/results/images/`
   - Metrics appended to `src/results/metrics.json`

5. **Expected duration:** 3-8 minutes (longer than FGSM due to iterative nature)

**Using custom directory:**
```bash
python demo_attack.py --attack pgd --eval_dir ../data/holdout
```

### Running All Attacks in Sequence

To test all attacks one after another:

```bash
cd src

# Run all three attacks
python demo_attack.py --attack patch
python demo_attack.py --attack fgsm
python demo_attack.py --attack pgd
```

**Total expected duration:** 10-25 minutes (depending on hardware)

### Verifying Results

After running all attacks, check the results:

1. **Metrics file:** `src/results/metrics.json`
   - Contains attack success rates, confidence shifts, and robustness scores for all attacks
   - Each attack's results are appended as a separate entry

2. **Visualizations:** `src/results/images/`
   - Comparison images showing original vs. adversarial examples
   - Each image shows similarity scores to the target caption

3. **Patch image:** `src/output/universal_patch.png`
   - The generated universal adversarial patch

### Understanding the Output

Each attack outputs:
- **Attack Success Rate (ASR)**: Percentage of images where the attack successfully increased similarity to target caption
- **Average Confidence Shift**: Mean change in similarity score
- **Robustness Score**: `1 - ASR` (higher is better for model robustness)

The target caption by default is `"a photo of a banana"`. You can change this in `src/config.py`.

## Research Context

This implementation demonstrates cross-modal adversarial attacks where:
- **Input Modality**: Images (visual)
- **Output Modality**: Text/Language (CLIP text embeddings)
- **Attack Goal**: Perturb images to maximize similarity with target caption

This reveals vulnerabilities in multimodal models where perturbations in one modality (vision) can manipulate outputs in another modality (language).

## License

See LICENSE file for details.
