#  Cross-Modal Adversarial Attacks  
### Universal Adversarial Patch Attack on CLIP (Vision–Language Model)

This project demonstrates a **Cross-Modal Targeted Universal Adversarial Patch Attack** on **CLIP**, a Vision–Language Model (VLM).  
The goal: create a small image patch that, when placed on *any* image, forces CLIP to output a specific incorrect caption — regardless of what the image actually contains.

This shows a fundamental weakness in multimodal AI systems:
- We perturb the **image**
- The model fails in the **language output**

This README will guide you through understanding, installing, and running the project.

---

##  Features

- ✔ Generates a **universal adversarial patch** (100×100 pixels)  
- ✔ Patch works on *any* image (even unseen images)  
- ✔ Fully **targeted attack** on CLIP  
- ✔ Clean evaluation with Attack Success Rate (ASR)  
- ✔ Before/after visual results saved automatically  
- ✔ Clear, descriptive logs for beginners & mentors  
- ✔ 100% self-contained — bring your own images  

---

##  What Is a “Universal Adversarial Patch”?

A **universal patch** is a small, learnable texture applied to images that causes a neural network to fail.  
In this project, the patch:

- Works on ANY image  
- Appears in ANY location  
- Forces CLIP to output a TARGET caption (default: `"a photo of a banana"`)

This is a powerful attack concept: **one patch fools the entire model**.

---

## Repository Structure
crossmodal-attacks/
│
├── data/
│ ├── images/ # training images (20–40 recommended)
│ └── holdout/ # evaluation images (10–20 unseen)
│
├── output/
│ ├── universal_patch.png # the learned adversarial patch
│ ├── example_0.png # side-by-side before/after comparisons
│ ├── example_1.png
│ └── ...
│
├── attack_demo.py # main script: patch training + evaluation
├── env.yml # conda environment file
├── requirements.txt # alternate pip dependency list
└── README.md # documentation 


---

## 🔧 Installation

### 1️. Clone the repository

```bash
git clone https://github.com/<your-username>/crossmodal-attacks
cd crossmodal-attacks

```
### 2. Install Dependencies
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Add images in data/images and data/holdout by running the following command in each directory :
for data/images : for ($i = 1; $i -le 30; $i++) {
    Invoke-WebRequest -Uri "https://picsum.photos/800/800?random=$i" -OutFile "img_$i.jpg"
}

For data/holdout : for ($i = 1; $i -le 10; $i++) {
    Invoke-WebRequest -Uri "https://picsum.photos/800/800?random=$i" -OutFile "img_$i.jpg"
}

### 4. Run the attack 

python demo_attack.py

### 5. Check the logs for Results

