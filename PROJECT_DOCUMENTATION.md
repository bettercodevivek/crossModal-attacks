# Cross-Modal Adversarial Attacks - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Purpose](#project-purpose)
3. [File Structure and Detailed Explanations](#file-structure-and-detailed-explanations)
4. [Attack Methods Explained](#attack-methods-explained)
5. [Understanding Attack Results](#understanding-attack-results)
6. [How Everything Works Together](#how-everything-works-together)

---

## Project Overview

This project implements a **cross-modal adversarial attack framework** designed to test the robustness of multimodal AI models, specifically CLIP (Contrastive Language-Image Pre-training). The framework demonstrates how small perturbations to images can manipulate text-based predictions in vision-language models, revealing vulnerabilities in cross-modal understanding.

### Key Concept: Cross-Modal Attacks

Traditional adversarial attacks typically perturb inputs to cause misclassification within the same modality (e.g., image perturbations causing image misclassification). This project focuses on **cross-modal attacks**, where:
- **Input Modality**: Images (visual data)
- **Output Modality**: Text/Language embeddings (CLIP text predictions)
- **Attack Goal**: Perturb images to maximize similarity with a target text caption

This reveals that multimodal models can be vulnerable to attacks that exploit the connection between different modalities (vision and language).

---

## Project Purpose

The project serves several purposes:

1. **Research**: Demonstrates cross-modal vulnerabilities in state-of-the-art vision-language models
2. **Robustness Testing**: Provides tools to evaluate how well models resist adversarial perturbations
3. **Security Analysis**: Helps identify potential security risks in multimodal AI systems
4. **Educational**: Shows implementation of various adversarial attack methods

The framework implements three different attack strategies, each with different characteristics and use cases, allowing comprehensive robustness evaluation.

---

## File Structure and Detailed Explanations

### Root Directory Files

#### `README.md`
- **Purpose**: User-facing documentation with setup instructions, usage examples, and quick start guide
- **Contents**: Installation steps, command-line usage, configuration options, and testing instructions
- **Audience**: End users who want to run the attacks

#### `LICENSE`
- **Purpose**: Legal license file specifying terms of use for the project

#### `PROJECT_DOCUMENTATION.md` (this file)
- **Purpose**: Comprehensive technical documentation explaining every component in detail
- **Contents**: Deep dive into architecture, algorithms, file purposes, and result interpretation

---

### Configuration and Core Files

#### `src/config.py`
- **Purpose**: Centralized configuration management for all attack parameters and system settings
- **Key Components**:
  - **Data Directories**: Paths to training images (`DATA_DIR`), evaluation images (`HOLDOUT_DIR`), and output directories
  - **Model Settings**: CLIP model name (`openai/clip-vit-base-patch32`) and image preprocessing size (224x224)
  - **Patch Attack Parameters**:
    - `PATCH_SIZE`: Size of the adversarial patch (100x100 pixels)
    - `PATCH_STEPS`: Number of optimization iterations (800 steps)
    - `PATCH_BATCH_SIZE`: Batch size for patch training (8 images)
    - `PATCH_LR`: Learning rate for patch optimization (0.1)
    - `PATCH_LOCATION`: Where to place patch ("random" or "topleft")
  - **FGSM Attack Parameters**:
    - `FGSM_EPSILON`: Maximum perturbation magnitude (0.03, meaning 3% of pixel value range)
  - **PGD Attack Parameters**:
    - `PGD_EPSILON`: Maximum perturbation budget (0.03)
    - `PGD_STEPS`: Number of iterative steps (40 iterations)
    - `PGD_ALPHA`: Step size per iteration (0.01)
  - **Target Configuration**:
    - `TARGET_TEXT`: The target caption to maximize similarity with (default: "a photo of a banana")
    - `CANDIDATE_TEXTS`: Alternative captions for evaluation comparison
  - **Utility Functions**:
    - `ensure_dirs()`: Creates necessary output directories if they don't exist

**Why it exists**: Separates configuration from code logic, making it easy to adjust attack parameters without modifying implementation files.

---

#### `src/utils.py`
- **Purpose**: Shared utility functions used across the entire project
- **Key Functions**:
  - `get_device()`: Automatically detects and returns available compute device (CUDA GPU if available, else CPU)
  - `load_image_paths(folder)`: Scans a directory and returns sorted list of image file paths (supports .jpg, .jpeg, .png, .bmp)
  - `get_image_preprocessor()`: Creates a PyTorch transform pipeline that resizes images to 224x224 and converts to tensor format (required by CLIP)
  - `open_and_preprocess(path, preprocess)`: Opens an image file, converts to RGB, and applies preprocessing transforms
  - `load_image_tensors(image_paths, device)`: Batch loads and preprocesses multiple images, returning a list of tensor objects
  - `tensor_to_image(tensor)`: Converts a PyTorch tensor back to numpy array format for visualization (handles channel ordering and value scaling)
  - `clamp_tensor(tensor, min_val, max_val)`: Clamps tensor values to valid range (typically [0, 1] for normalized images)

**Why it exists**: Provides reusable functionality, reducing code duplication and ensuring consistent image handling across all modules.

---

#### `src/demo_attack.py`
- **Purpose**: Main entry point script that orchestrates the entire attack pipeline
- **Key Components**:
  - **Command-Line Interface**: Parses arguments to select attack type (`--attack patch/fgsm/pgd`) and optional data directories
  - **Model Loading**: `load_clip_model()` loads the CLIP model and processor, handling device placement
  - **Attack Orchestration Functions**:
    - `run_patch_attack()`: Handles universal patch attack workflow
    - `run_fgsm_attack()`: Handles FGSM attack workflow
    - `run_pgd_attack()`: Handles PGD attack workflow
  - **Results Management**: `save_results()` appends attack results to JSON file
  - **Main Function**: Coordinates the entire pipeline:
    1. Parse command-line arguments
    2. Load CLIP model
    3. Load image paths from specified directories
    4. Execute selected attack
    5. Compute evaluation metrics
    6. Save results to JSON
    7. Generate visualization comparisons

**Workflow for Each Attack**:
1. Load model and images
2. Initialize attack object
3. Generate adversarial examples
4. Compute similarity scores (original vs. adversarial)
5. Calculate metrics (ASR, confidence shift, robustness)
6. Save results and generate visualizations

**Why it exists**: Provides a unified interface to run all attack types with consistent evaluation and output formatting.

---

### Attack Implementation Files

#### `src/attacks/__init__.py`
- **Purpose**: Package initialization file that exports all attack classes
- **Exports**: `PatchAttack`, `FGSMAttack`, `PGDAttack`
- **Why it exists**: Makes attacks easily importable with `from attacks import PatchAttack, FGSMAttack, PGDAttack`

---

#### `src/attacks/patch_attack.py`
- **Purpose**: Implements the Universal Adversarial Patch attack
- **Class**: `PatchAttack`
- **How It Works**:
  1. **Initialization**: 
     - Stores CLIP model, processor, target text, and device
     - Precomputes the target text embedding (normalized) for efficiency
  2. **Patch Generation** (`generate_patch()`):
     - Creates a learnable patch parameter (random initialization)
     - Uses Adam optimizer to iteratively improve the patch
     - For each training step:
       - Samples a random batch of training images
       - Applies the current patch to images (at random or fixed location)
       - Computes CLIP image embeddings for patched images
       - Calculates cosine similarity with target text embedding
       - Maximizes similarity via gradient descent (minimizes negative similarity)
       - Clamps patch values to valid range [0, 1]
     - Returns the optimized patch tensor
  3. **Patch Application** (`apply_patch()`):
     - Takes images and the generated patch
     - Overlays patch onto images at specified location
     - Returns patched images

**Key Characteristics**:
- **Universal**: One patch works on any image (unlike image-specific attacks)
- **Transferable**: Patch trained on one set of images can fool model on different images
- **Visible**: The patch is a visible overlay on the image (unlike subtle pixel perturbations)
- **Training Required**: Needs training images to optimize the patch

**Mathematical Objective**:
- Maximize: `cosine_similarity(CLIP_image_embedding(patched_image), CLIP_text_embedding(target_caption))`
- Uses gradient descent to find patch values that maximize this similarity

**Why it exists**: Demonstrates that a single universal patch can fool CLIP across multiple images, showing a practical attack vector.

---

#### `src/attacks/fgsm_attack.py`
- **Purpose**: Implements the Fast Gradient Sign Method (FGSM) attack
- **Class**: `FGSMAttack`
- **How It Works**:
  1. **Initialization**: 
     - Stores model, processor, target text, device
     - Precomputes normalized target text embedding
  2. **Attack Generation** (`attack()`):
     - Enables gradients on input images
     - Forward pass: Computes CLIP image embeddings
     - Computes cosine similarity with target text
     - Backward pass: Calculates gradients of similarity with respect to image pixels
     - **FGSM Formula**: `x_adv = x + epsilon * sign(∇_x similarity)`
     - Clamps result to valid image range [0, 1]

**Key Characteristics**:
- **Single-Step**: Only one gradient computation (very fast)
- **Image-Specific**: Each image gets its own perturbation
- **Subtle**: Perturbations are small (epsilon = 0.03 = 3% of pixel range)
- **Fast**: Computationally efficient compared to iterative methods

**Mathematical Formula**:
```
x_adv = x + ε * sign(∇_x L(x, target_text))
```
Where:
- `x` = original image
- `ε` = epsilon (perturbation budget)
- `L` = loss function (negative similarity)
- `sign()` = sign function (returns +1 or -1)

**Why it exists**: Provides a fast baseline attack method, demonstrating that even simple single-step attacks can manipulate cross-modal predictions.

---

#### `src/attacks/pgd_attack.py`
- **Purpose**: Implements the Projected Gradient Descent (PGD) attack
- **Class**: `PGDAttack`
- **How It Works**:
  1. **Initialization**: Same as FGSM (precomputes target embedding)
  2. **Attack Generation** (`attack()`):
     - Starts with original image as initial adversarial example
     - **Iterative Process** (repeats for `num_steps` iterations):
       - Enables gradients on current adversarial image
       - Forward pass: Computes similarity with target
       - Backward pass: Gets gradients
       - **Update Step**: `x_adv = x_adv + alpha * sign(∇_x similarity)`
       - **Projection**: Clamps to epsilon-ball around original: `clip(x_adv, x - ε, x + ε)`
       - Clamps to valid image range [0, 1]
       - Re-enables gradients for next iteration
     - Returns final adversarial image

**Key Characteristics**:
- **Iterative**: Multiple gradient steps (typically 40 iterations)
- **Stronger**: Generally more effective than FGSM due to iterative refinement
- **Bounded**: Perturbations constrained to epsilon-ball (L∞ norm)
- **Image-Specific**: Each image gets custom perturbation
- **Slower**: More computationally expensive than FGSM

**Mathematical Algorithm**:
```
x_adv^(0) = x
for t = 1 to num_steps:
    x_adv^(t) = x_adv^(t-1) + α * sign(∇_x L(x_adv^(t-1), target))
    x_adv^(t) = clip(x_adv^(t), x - ε, x + ε)  // Project to epsilon-ball
    x_adv^(t) = clip(x_adv^(t), 0, 1)  // Valid image range
```

**Why it exists**: Provides a stronger attack baseline, showing that iterative methods can be more effective at finding adversarial examples.

---

### Evaluation Module Files

#### `src/evaluation/__init__.py`
- **Purpose**: Package initialization for evaluation module
- **Why it exists**: Makes evaluation functions easily importable

---

#### `src/evaluation/metrics.py`
- **Purpose**: Defines and computes all evaluation metrics for attack effectiveness
- **Key Functions**:
  1. **`compute_similarity(model, processor, images, target_text, device)`**:
     - Computes cosine similarity between image embeddings and target text embedding
     - Returns array of similarity scores (one per image)
     - Uses normalized embeddings for cosine similarity calculation
  2. **`compute_asr(original_similarities, adversarial_similarities)`**:
     - **Attack Success Rate (ASR)**: Percentage of images where attack succeeded
     - Success defined as: `adversarial_similarity > original_similarity`
     - Formula: `ASR = (number of successful attacks) / (total number of images)`
     - Returns value between 0.0 and 1.0 (0% to 100%)
  3. **`compute_confidence_shift(original_similarities, adversarial_similarities)`**:
     - Measures average change in similarity score
     - Formula: `confidence_shift = mean(adversarial_similarity - original_similarity)`
     - Positive values indicate attack increased similarity (success)
     - Negative values indicate attack decreased similarity (failure)
  4. **`compute_robustness_score(asr)`**:
     - Inverse of ASR: `R = 1 - ASR`
     - Higher values indicate better model robustness
     - Range: 0.0 (completely vulnerable) to 1.0 (completely robust)
  5. **`compute_all_metrics(...)`**:
     - Convenience function that computes all metrics at once
     - Returns dictionary with ASR, confidence shift, robustness score, and raw similarity arrays

**Why it exists**: Provides standardized metrics for comparing attack effectiveness and model robustness across different attack methods.

---

#### `src/evaluation/robustness_evaluator.py`
- **Purpose**: High-level evaluation pipeline for batch processing and result management
- **Class**: `RobustnessEvaluator`
- **Key Methods**:
  - **`__init__(model, processor, device)`**: Initializes evaluator with CLIP model
  - **`evaluate_attack(attack, image_paths, target_text, ...)`**:
    - Loads images from file paths
    - Processes images in batches (for memory efficiency)
    - Applies attack to generate adversarial examples
    - Computes all evaluation metrics
    - Optionally saves results to JSON file
    - Returns results dictionary and image tensors

**Why it exists**: Provides a reusable evaluation pipeline that can be used for systematic robustness testing across different attacks and datasets.

---

### Visualization Module Files

#### `src/visualization/__init__.py`
- **Purpose**: Package initialization for visualization module
- **Why it exists**: Makes visualization functions easily importable

---

#### `src/visualization/visualize_results.py`
- **Purpose**: Generates side-by-side comparison visualizations of original vs. adversarial images
- **Key Functions**:
  - **`visualize_attack_results(...)`**:
    - Creates side-by-side comparison images (original on left, adversarial on right)
    - Displays similarity scores for both images
    - Saves comparison images to `results/images/` directory
    - Limits to `max_images` (default 8) to avoid excessive output
    - Uses matplotlib for image generation
  - **`save_comparison_images(...)`**: Alias for `visualize_attack_results()`

**Output Format**:
- Each comparison image shows:
  - Left: Original image with similarity score to target caption
  - Right: Adversarial image with similarity score to target caption
- Filenames: `comparison_000.png`, `comparison_001.png`, etc.

**Why it exists**: Provides visual evidence of attack effectiveness, making it easy to see how attacks change model predictions.

---

### Data Directories

#### `data/images/`
- **Purpose**: Training images for patch attack
- **Usage**: Used by patch attack to optimize the universal patch
- **Format**: Any standard image format (.jpg, .jpeg, .png, .bmp)
- **Note**: Not required for FGSM or PGD attacks

#### `data/holdout/`
- **Purpose**: Evaluation/test images for all attacks
- **Usage**: Used to evaluate attack effectiveness on unseen images
- **Format**: Any standard image format
- **Note**: Should be different from training images to properly evaluate generalization

---

### Output Directories

#### `src/output/`
- **Purpose**: Stores generated artifacts
- **Contents**: 
  - `universal_patch.png`: The generated universal adversarial patch (from patch attack)

#### `src/results/`
- **Purpose**: Stores evaluation results
- **Contents**:
  - `metrics.json`: JSON file containing all attack results (ASR, confidence shifts, robustness scores)
  - `images/`: Directory containing comparison visualizations (`comparison_*.png`)

---

## Attack Methods Explained

### 1. Universal Adversarial Patch Attack

#### Overview
The patch attack generates a **universal** adversarial patch—a single image patch that, when overlaid on any image, can manipulate CLIP's predictions toward a target caption.

#### How It Works
1. **Training Phase**:
   - Initializes a random patch (e.g., 100x100 pixels)
   - Iteratively optimizes the patch over multiple training images
   - Uses gradient descent to maximize similarity with target caption
   - After 800 optimization steps, the patch is ready

2. **Application Phase**:
   - The trained patch can be applied to any new image
   - Patch is overlaid at a specified location (random or top-left)
   - CLIP processes the patched image and outputs higher similarity to target caption

#### Algorithm Details
```
Initialize patch P (random values)
For each training step:
    Sample batch of training images
    Apply patch P to each image
    Compute CLIP embeddings for patched images
    Calculate similarity with target text embedding
    Compute loss = -similarity (to maximize similarity)
    Update patch: P = P - lr * ∇_P loss
    Clamp P to [0, 1]
```

#### Advantages
- **Universal**: One patch works on many images
- **Practical**: Can be printed and physically applied to images
- **Transferable**: Works across different image types

#### Limitations
- **Visible**: The patch is clearly visible (not stealthy)
- **Training Required**: Needs training images and computation time
- **Location Dependent**: Effectiveness may vary with patch placement

#### Use Cases
- Testing robustness to physical adversarial examples
- Demonstrating universal vulnerabilities
- Security analysis of vision systems

---

### 2. FGSM (Fast Gradient Sign Method) Attack

#### Overview
FGSM is a **single-step** attack that perturbs images using the sign of the gradient. It's fast but generally less effective than iterative methods.

#### How It Works
1. **Forward Pass**: Compute CLIP embeddings for original images
2. **Similarity Calculation**: Calculate cosine similarity with target text
3. **Gradient Computation**: Compute gradients of similarity with respect to image pixels
4. **Perturbation**: Add epsilon-scaled sign of gradient to original image
5. **Clipping**: Ensure pixel values remain in valid range [0, 1]

#### Algorithm
```
x_adv = x + ε * sign(∇_x similarity(x, target_text))
x_adv = clip(x_adv, 0, 1)
```

#### Advantages
- **Fast**: Single gradient computation (very efficient)
- **Simple**: Easy to understand and implement
- **Baseline**: Good starting point for attack evaluation

#### Limitations
- **Less Effective**: Single step may not find optimal perturbations
- **Image-Specific**: Must compute for each image separately
- **Subtle**: Perturbations are small and may not always succeed

#### Use Cases
- Quick robustness testing
- Baseline comparison for stronger attacks
- Real-time attack scenarios where speed matters

---

### 3. PGD (Projected Gradient Descent) Attack

#### Overview
PGD is an **iterative** attack that performs multiple gradient steps with projection, making it stronger than FGSM.

#### How It Works
1. **Initialization**: Start with original image as initial adversarial example
2. **Iterative Refinement** (repeat for N steps):
   - Compute gradient of similarity with respect to current adversarial image
   - Take a small step in gradient direction: `x_adv = x_adv + alpha * sign(grad)`
   - Project back to epsilon-ball: `clip(x_adv, x - ε, x + ε)`
   - Ensure valid pixel range: `clip(x_adv, 0, 1)`
3. **Result**: Final adversarial image after all iterations

#### Algorithm
```
x_adv^(0) = x
For t = 1 to num_steps:
    grad = ∇_x similarity(x_adv^(t-1), target_text)
    x_adv^(t) = x_adv^(t-1) + α * sign(grad)
    x_adv^(t) = clip(x_adv^(t), x - ε, x + ε)  // Project to epsilon-ball
    x_adv^(t) = clip(x_adv^(t), 0, 1)  // Valid range
```

#### Advantages
- **Stronger**: Iterative refinement finds better adversarial examples
- **Bounded**: Perturbations constrained to epsilon-ball (L∞ norm)
- **Flexible**: Can adjust number of steps and step size

#### Limitations
- **Slower**: Multiple iterations require more computation
- **Image-Specific**: Must compute for each image
- **Hyperparameter Sensitive**: Performance depends on epsilon, steps, and alpha

#### Use Cases
- Comprehensive robustness evaluation
- Finding strongest possible attacks within perturbation budget
- Benchmarking model defenses

---

## Understanding Attack Results

### Metrics Explained

#### 1. Attack Success Rate (ASR)

**Definition**: Percentage of images where the attack successfully increased similarity to the target caption.

**Formula**: 
```
ASR = (number of images where adv_similarity > orig_similarity) / total_images
```

**Interpretation**:
- **ASR = 0.0 (0%)**: Attack failed on all images (model is robust)
- **ASR = 0.5 (50%)**: Attack succeeded on half the images
- **ASR = 1.0 (100%)**: Attack succeeded on all images (model is vulnerable)

**What It Means**:
- **High ASR (e.g., > 0.8)**: Model is vulnerable to this attack type
- **Low ASR (e.g., < 0.2)**: Model is relatively robust to this attack type
- **ASR = 0.0**: Attack actually decreased similarity (attack failed)

**Example from Results**:
```json
{
  "attack": "FGSMAttack",
  "asr": 0.0,
  ...
}
```
This means FGSM attack failed to increase similarity on any images (0% success rate).

---

#### 2. Average Confidence Shift

**Definition**: Average change in similarity score between adversarial and original images.

**Formula**:
```
confidence_shift = mean(adversarial_similarity - original_similarity)
```

**Interpretation**:
- **Positive values**: Attack increased similarity (successful attack)
  - Example: `+0.42` means similarity increased by 0.42 on average
- **Negative values**: Attack decreased similarity (failed attack)
  - Example: `-0.035` means similarity decreased by 0.035 on average
- **Zero**: No change on average

**What It Means**:
- **Large positive shift (e.g., > 0.3)**: Attack is very effective at manipulating predictions
- **Small positive shift (e.g., 0.01-0.1)**: Attack has mild effect
- **Negative shift**: Attack is counterproductive (makes model less likely to predict target)

**Example from Results**:
```json
{
  "attack": "FGSMAttack",
  "avg_confidence_shift": -0.03524542227387428,
  ...
}
```
This means FGSM attack actually decreased similarity by 0.035 on average, indicating the attack failed.

---

#### 3. Robustness Score

**Definition**: Inverse of ASR, measuring how robust the model is to the attack.

**Formula**:
```
robustness_score = 1 - ASR
```

**Interpretation**:
- **R = 1.0**: Model is completely robust (ASR = 0%)
- **R = 0.5**: Model is moderately robust (ASR = 50%)
- **R = 0.0**: Model is completely vulnerable (ASR = 100%)

**What It Means**:
- **High robustness (R > 0.8)**: Model resists attacks well
- **Low robustness (R < 0.2)**: Model is vulnerable to attacks
- This metric is simply the complement of ASR, providing an alternative perspective

**Example from Results**:
```json
{
  "attack": "FGSMAttack",
  "asr": 0.0,
  "robustness_score": 1.0,
  ...
}
```
This means the model showed perfect robustness (100%) to FGSM attacks.

---

### Interpreting Complete Results

#### Example Result Entry
```json
{
  "attack": "PatchAttack",
  "asr": 0.84,
  "avg_confidence_shift": 0.42,
  "robustness_score": 0.16,
  "num_images": 10
}
```

**Analysis**:
- **ASR = 0.84 (84%)**: Patch attack succeeded on 84% of images
- **Confidence Shift = +0.42**: On average, similarity increased by 0.42 (significant increase)
- **Robustness = 0.16 (16%)**: Model is only 16% robust (84% vulnerable)
- **Conclusion**: Patch attack is highly effective; model is vulnerable

#### Comparing Different Attacks

**Scenario 1: FGSM Results**
```json
{
  "attack": "FGSMAttack",
  "asr": 0.0,
  "avg_confidence_shift": -0.035,
  "robustness_score": 1.0
}
```
- **Interpretation**: FGSM failed completely; model is robust to FGSM
- **Possible Reasons**: 
  - Epsilon too small (0.03 may be insufficient)
  - Single-step attack not strong enough
  - Model has some inherent robustness to small perturbations

**Scenario 2: PGD Results**
```json
{
  "attack": "PGDAttack",
  "asr": 0.70,
  "avg_confidence_shift": 0.25,
  "robustness_score": 0.30
}
```
- **Interpretation**: PGD is moderately effective; model shows some vulnerability
- **Comparison**: PGD is stronger than FGSM (iterative vs. single-step)

**Scenario 3: Patch Results**
```json
{
  "attack": "PatchAttack",
  "asr": 0.90,
  "avg_confidence_shift": 0.50,
  "robustness_score": 0.10
}
```
- **Interpretation**: Patch attack is highly effective; model is very vulnerable
- **Note**: Patch attack is different—it's universal and visible, so high success rate is expected

---

### What Results Mean for Model Security

#### High Vulnerability (ASR > 0.7)
- **Implication**: Model can be easily fooled by adversarial examples
- **Risk**: High security risk in production systems
- **Action**: Consider implementing defenses (adversarial training, input validation, etc.)

#### Moderate Vulnerability (0.3 < ASR < 0.7)
- **Implication**: Model has some robustness but can still be attacked
- **Risk**: Moderate security risk
- **Action**: Monitor and consider additional security measures

#### Low Vulnerability (ASR < 0.3)
- **Implication**: Model is relatively robust to tested attacks
- **Risk**: Lower security risk (but test more attack types)
- **Action**: Continue monitoring, test with stronger attacks

#### Negative Confidence Shift
- **Implication**: Attack actually made predictions worse (attack failed)
- **Interpretation**: Model may have some inherent robustness, or attack parameters need tuning
- **Action**: Try different attack parameters or stronger attacks

---

### Understanding Visualization Results

The comparison images in `results/images/` show:

1. **Original Image** (left side):
   - Shows the unmodified image
   - Displays similarity score to target caption (e.g., "a photo of a banana")
   - Example: "Similarity to 'a photo of a banana': 0.123"

2. **Adversarial Image** (right side):
   - Shows the attacked image (with patch/perturbations)
   - Displays new similarity score
   - Example: "Similarity to 'a photo of a banana': 0.543"

3. **What to Look For**:
   - **Successful Attack**: Adversarial similarity > Original similarity
   - **Failed Attack**: Adversarial similarity < Original similarity
   - **Visual Changes**: 
     - Patch attack: Visible patch overlay
     - FGSM/PGD: May be subtle (look for slight color shifts or noise)

---

## How Everything Works Together

### Complete Attack Pipeline

1. **User Runs Command**:
   ```bash
   python demo_attack.py --attack pgd
   ```

2. **Initialization** (`demo_attack.py`):
   - Parses command-line arguments
   - Loads configuration from `config.py`
   - Detects device (GPU/CPU) via `utils.get_device()`
   - Loads CLIP model and processor

3. **Data Loading** (`demo_attack.py` + `utils.py`):
   - Scans `data/holdout/` for images using `load_image_paths()`
   - Loads and preprocesses images using `load_image_tensors()`
   - Converts images to tensors ready for CLIP

4. **Attack Execution** (`demo_attack.py` + attack module):
   - Initializes attack object (e.g., `PGDAttack`)
   - Attack object precomputes target text embedding
   - Calls `attack.attack()` method:
     - Iteratively perturbs images
     - Computes gradients
     - Updates adversarial examples
   - Returns adversarial image tensors

5. **Evaluation** (`demo_attack.py` + `evaluation/metrics.py`):
   - Computes similarity scores for original images
   - Computes similarity scores for adversarial images
   - Calculates ASR, confidence shift, robustness score
   - Prepares results dictionary

6. **Results Saving** (`demo_attack.py`):
   - Appends results to `results/metrics.json`
   - Prints metrics to console

7. **Visualization** (`demo_attack.py` + `visualization/visualize_results.py`):
   - Generates side-by-side comparisons
   - Saves images to `results/images/`
   - Shows similarity scores on images

### Data Flow

```
User Input (CLI)
    ↓
demo_attack.py (orchestration)
    ↓
config.py (parameters)
    ↓
utils.py (image loading, preprocessing)
    ↓
CLIP Model (embeddings)
    ↓
Attack Module (fgsm/pgd/patch)
    ↓
Adversarial Images
    ↓
evaluation/metrics.py (compute metrics)
    ↓
results/metrics.json (save results)
    ↓
visualization/visualize_results.py (generate images)
    ↓
results/images/ (save visualizations)
```

### Key Design Principles

1. **Modularity**: Each component (attacks, evaluation, visualization) is separate and reusable
2. **Configuration**: All parameters centralized in `config.py`
3. **Consistency**: All attacks use same evaluation metrics and output format
4. **Extensibility**: Easy to add new attack methods or evaluation metrics
5. **Usability**: Simple command-line interface for running attacks

---

## Summary

This project provides a comprehensive framework for testing cross-modal adversarial attacks on CLIP. It implements three attack methods (Patch, FGSM, PGD), each with different characteristics and use cases. The framework includes robust evaluation metrics, visualization tools, and a clean architecture that makes it easy to understand, use, and extend.

The results help researchers and practitioners understand:
- How vulnerable multimodal models are to cross-modal attacks
- Which attack methods are most effective
- How to measure and compare model robustness
- What security implications exist for production systems

By providing detailed metrics and visualizations, the framework enables informed decisions about model security and robustness improvements.

