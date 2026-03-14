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

The framework requires image data to run attacks. Follow these steps to set up your data directories and download test images.

#### 4.1: Create Data Directory Structure

Create the required directory structure in your project root:

**On Windows (PowerShell):**
```powershell
# Navigate to project root
cd crossModal-attacks

# Create directories
mkdir data
mkdir data\images
mkdir data\holdout
```

**On Windows (Command Prompt):**
```cmd
cd crossModal-attacks
mkdir data
mkdir data\images
mkdir data\holdout
```

**On Linux/Mac:**
```bash
# Navigate to project root
cd crossModal-attacks

# Create directories
mkdir -p data/images
mkdir -p data/holdout
```

The expected directory structure should look like:
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

#### 4.2: Download Test Images

You need images in two directories:
- **`data/images/`**: Training images for patch attack (recommended: 20-50 images)
- **`data/holdout/`**: Evaluation images for all attacks (recommended: 10-30 images)

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`

##### Quick Download: Simple Python Script (Easiest Method) ⭐

**The project includes a ready-to-use download script!** Just run it from the project root:

```bash
# Install requests if not already installed
pip install requests

# Run the download script (from project root)
python download_images.py
```

This single command will:
- ✅ Automatically create `data/images/` and `data/holdout/` directories
- ✅ Download 30 diverse images to `data/images/` for training
- ✅ Download 10 diverse images to `data/holdout/` for evaluation
- ✅ Save images as `img_1.jpg`, `img_2.jpg`, etc.
- ✅ Show progress for each download

**That's it!** After running this command, you're ready to run the attacks.

**Note:** The script uses Unsplash Source API (free, no API key required) to download random images with diverse keywords (dogs, cats, nature, buildings, etc.) for comprehensive testing.

**Alternative: One-Line Command (Using curl/wget)**

If you prefer command-line tools:

**On Linux/Mac:**
```bash
# Download 30 training images
mkdir -p data/images
for i in {1..30}; do
    curl -L "https://source.unsplash.com/800x600/?random" -o "data/images/img_$i.jpg"
    echo "Downloaded img_$i.jpg"
done

# Download 10 evaluation images
mkdir -p data/holdout
for i in {1..10}; do
    curl -L "https://source.unsplash.com/800x600/?random" -o "data/holdout/img_$i.jpg"
    echo "Downloaded img_$i.jpg"
done
```

**On Windows (PowerShell):**
```powershell
# Download 30 training images
mkdir -p data/images
1..30 | ForEach-Object {
    Invoke-WebRequest -Uri "https://source.unsplash.com/800x600/?random" -OutFile "data/images/img_$_.jpg"
    Write-Host "Downloaded img_$_.jpg"
}

# Download 10 evaluation images
mkdir -p data/holdout
1..10 | ForEach-Object {
    Invoke-WebRequest -Uri "https://source.unsplash.com/800x600/?random" -OutFile "data/holdout/img_$_.jpg"
    Write-Host "Downloaded img_$_.jpg"
}
```

Here are additional methods to obtain test images:

##### Method 1: Download from Free Image Websites (Recommended for Quick Testing)

1. **Unsplash** (https://unsplash.com/):
   - Search for diverse images (animals, objects, scenes, etc.)
   - Download images directly to your computer
   - Move them to the appropriate directories

2. **Pexels** (https://www.pexels.com/):
   - Browse and download free stock photos
   - Good variety of subjects for testing

3. **Pixabay** (https://pixabay.com/):
   - Free images with various categories
   - Download and organize into folders

**Quick Setup Steps:**
- Download 20-50 diverse images (various objects, animals, scenes)
- Place them in `data/images/` for patch attack training
- Download 10-30 different images
- Place them in `data/holdout/` for evaluation

##### Method 2: Use Python Script to Download Images

Create a simple download script to fetch images automatically:

```python
# download_images.py
import requests
import os
from urllib.parse import urlparse

# Example: Download images from URLs
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    # Add more URLs
]

def download_image(url, save_dir):
    """Download an image from URL and save to directory."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = f"image_{hash(url)}.jpg"
        
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Create directories
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/holdout", exist_ok=True)

# Download training images
for url in image_urls[:30]:  # First 30 for training
    download_image(url, "data/images")

# Download evaluation images
for url in image_urls[30:]:  # Remaining for evaluation
    download_image(url, "data/holdout")
```

**Note:** You'll need to install `requests`: `pip install requests`

##### Method 3: Use Public Datasets

1. **COCO Dataset** (https://cocodataset.org/):
   - Large-scale dataset with diverse images
   - Download validation set for testing
   - Extract images to your directories

2. **ImageNet** (http://www.image-net.org/):
   - Large image classification dataset
   - Download sample images for testing

3. **Kaggle Datasets**:
   - Search for "image classification" or "object detection" datasets
   - Download and extract images

##### Method 4: Use Your Own Images

If you have your own image collection:
1. Copy images from your computer
2. Ensure they're in supported formats (`.jpg`, `.jpeg`, `.png`, `.bmp`)
3. Place training images in `data/images/`
4. Place test images in `data/holdout/`

**Tips:**
- Use diverse images (different objects, scenes, lighting conditions)
- Ensure images are reasonably sized (will be resized to 224x224 by CLIP)
- Avoid corrupted or extremely small images

#### 4.3: Verify Image Setup

After downloading images, verify your setup:

**On Windows (PowerShell):**
```powershell
# Count images in each directory
(Get-ChildItem data\images\*.jpg, data\images\*.png, data\images\*.jpeg, data\images\*.bmp).Count
(Get-ChildItem data\holdout\*.jpg, data\holdout\*.png, data\holdout\*.jpeg, data\holdout\*.bmp).Count
```

**On Linux/Mac:**
```bash
# Count images in each directory
ls -1 data/images/*.{jpg,jpeg,png,bmp} 2>/dev/null | wc -l
ls -1 data/holdout/*.{jpg,jpeg,png,bmp} 2>/dev/null | wc -l
```

**Minimum Requirements:**
- `data/images/`: At least 10 images (more is better for patch attack)
- `data/holdout/`: At least 5 images (more is better for evaluation)

**Recommended:**
- `data/images/`: 20-50 images for better patch attack performance
- `data/holdout/`: 10-30 images for comprehensive evaluation

#### 4.4: Rename Images (Optional)

For better organization, you can rename images sequentially:

**On Windows (PowerShell):**
```powershell
# Rename images in data/images/
$i = 1
Get-ChildItem data\images\*.jpg, data\images\*.png | ForEach-Object {
    $newName = "img_$i.jpg"
    Rename-Item $_.FullName -NewName $newName
    $i++
}

# Rename images in data/holdout/
$i = 1
Get-ChildItem data\holdout\*.jpg, data\holdout\*.png | ForEach-Object {
    $newName = "img_$i.jpg"
    Rename-Item $_.FullName -NewName $newName
    $i++
}
```

**On Linux/Mac:**
```bash
# Rename images in data/images/
cd data/images
i=1
for file in *.jpg *.jpeg *.png; do
    [ -f "$file" ] && mv "$file" "img_$i.jpg" && ((i++))
done

# Rename images in data/holdout/
cd ../holdout
i=1
for file in *.jpg *.jpeg *.png; do
    [ -f "$file" ] && mv "$file" "img_$i.jpg" && ((i++))
done
```

**Note:** The framework works with any image filenames - renaming is optional and only for organization.

#### 4.5: Path Configuration

**Important Notes:**
- Images should be placed in directories relative to the **project root** (not `src/` directory)
- Default paths in `config.py` are: `../data/images` and `../data/holdout` (relative to `src/`)
- When running from `src/` directory, paths resolve correctly
- If you use custom directories, specify them via command-line arguments:
  ```bash
  python demo_attack.py --attack patch --train_dir ../data/images --eval_dir ../data/holdout
  ```

**Troubleshooting:**
- If you get "No images found" error, check:
  1. Directory paths are correct
  2. Images are in supported formats (`.jpg`, `.jpeg`, `.png`, `.bmp`)
  3. You're running commands from the `src/` directory
  4. Use absolute paths if relative paths don't work

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
