# attack_demo.py
import os
import random
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------- CONFIG ----------
DATA_DIR = "data/images"       # training images
HOLDOUT_DIR = "data/holdout"   # evaluation images (unseen)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
PATCH_SIZE = 100
PATCH_LOCATION = "random"      # "topleft" or "random"
STEPS = 800
LR = 0.1
TARGET_TEXT = "a photo of a banana"
CANDIDATE_TEXTS = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a person",
    "a photo of a car",
    "a photo of a banana"
]
SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)
# ---------------------------

print("\n===============================")
print("CROSS-MODAL ADVERSARIAL ATTACK DEMO")
print("===============================\n")

print(f" Device selected: {DEVICE.upper()}")
print("This script will generate a UNIVERSAL adversarial patch that can fool CLIP.")
print("Once trained, applying this patch on ANY image will cause CLIP to misclassify it")
print(f"as the target caption: \"{TARGET_TEXT}\".\n")

print(" This demonstrates a **cross-modal attack** because:")
print("- We perturb the IMAGE modality")
print("- And it causes errors in the LANGUAGE output of CLIP.\n")

print("Loading model and processor... (first time may take a minute)\n")

# load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# helper to load paths
def load_image_paths(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for e in exts:
        files += glob(os.path.join(folder, e))
    return sorted(files)

train_paths = load_image_paths(DATA_DIR)
hold_paths = load_image_paths(HOLDOUT_DIR)

if len(train_paths) == 0:
    raise SystemExit("❌ ERROR: No images found in data/images. Add images and re-run.")

print(f" Training images loaded: {len(train_paths)}")
print(f" Holdout images loaded: {len(hold_paths)}\n")

print("📌 Candidate captions CLIP will choose from:")
for i, cap in enumerate(CANDIDATE_TEXTS, 1):
    print(f"   {i}. {cap}")
print(f"\n Target caption index: {CANDIDATE_TEXTS.index(TARGET_TEXT)}")
print(f" We will FORCE CLIP to choose: \"{TARGET_TEXT}\"\n")

# open + preprocess image
def open_and_preprocess(path):
    img = Image.open(path).convert("RGB")
    return preprocess(img)

train_tensors = [open_and_preprocess(p) for p in train_paths]
hold_tensors = [open_and_preprocess(p) for p in hold_paths] if len(hold_paths) > 0 else []

def get_batch_random(batch_size=BATCH_SIZE):
    items = random.sample(train_tensors, min(batch_size, len(train_tensors)))
    return torch.stack(items, dim=0).to(DEVICE)

# create learnable patch
patch = nn.Parameter(torch.rand(3, PATCH_SIZE, PATCH_SIZE, device=DEVICE) * 0.5)
opt = torch.optim.Adam([patch], lr=LR)

# compute text embeddings
with torch.no_grad():
    text_inputs = processor(text=CANDIDATE_TEXTS, return_tensors="pt", padding=True).to(DEVICE)
    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

target_index = CANDIDATE_TEXTS.index(TARGET_TEXT)
print("\n===============================")
print(" TRAINING THE ADVERSARIAL PATCH")
print("===============================\n")
print("We will optimize a 100x100 patch so that:")
print(f"- When placed on ANY image → CLIP believes it is \"{TARGET_TEXT}\".")
print("- Training uses random images and random patch placements.")
print("- Every iteration increases similarity to the target caption.\n")

pbar = tqdm(range(STEPS))
for step in pbar:
    model.zero_grad()
    imgs = get_batch_random()
    patched = imgs.clone()

    # apply patch
    for i in range(patched.size(0)):
        if PATCH_LOCATION == "random":
            x = random.randint(0, patched.shape[-1] - PATCH_SIZE)
            y = random.randint(0, patched.shape[-2] - PATCH_SIZE)
        else:
            x = y = 0
        patched[i, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch

    inputs = {"pixel_values": patched}
    image_embs = model.get_image_features(**inputs)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    sims = image_embs @ text_embs.T
    target_sim = sims[:, target_index].mean()

    loss = -target_sim + 0.01 * ((patch - patch.detach()).pow(2).mean())
    loss.backward()
    opt.step()

    with torch.no_grad():
        patch.clamp_(0, 1)

    if step % 50 == 0:
        pbar.set_description(f"Step {step} | Loss: {loss.item():.4f} | TargetSim: {target_sim.item():.4f}")

print("\n Training complete! Saving patch...")
patch_img = (patch.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
plt.imsave(os.path.join(SAVE_DIR, "universal_patch.png"), patch_img)
print(" Saved adversarial patch to:", os.path.join(SAVE_DIR, "universal_patch.png"))

# eval helper
def predict_top1_texts(image_tensors):
    with torch.no_grad():
        inputs = {"pixel_values": image_tensors}
        im_embs = model.get_image_features(**inputs)
        im_embs = im_embs / im_embs.norm(dim=-1, keepdim=True)
        sims = (im_embs @ text_embs.T).cpu().numpy()
        return sims.argmax(axis=1), sims

print("\n===============================")
print(" EVALUATION ON UNSEEN HOLDOUT IMAGES")
print("===============================\n")

if len(hold_tensors) == 0:
    print(" No holdout images found — skipping evaluation.")
else:
    print(f"Testing on {len(hold_tensors)} images the model NEVER saw during training.\n")

    batch = torch.stack(hold_tensors, dim=0).to(DEVICE)

    print(" Getting CLIP predictions BEFORE applying patch...")
    orig_top1, _ = predict_top1_texts(batch)

    print(" Applying patch to holdout images...")
    patched_batch = batch.clone()
    for i in range(patched_batch.size(0)):
        x = random.randint(0, 224 - PATCH_SIZE)
        y = random.randint(0, 224 - PATCH_SIZE)
        patched_batch[i, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch.detach()

    print(" Getting CLIP predictions AFTER applying patch...")
    patched_top1, _ = predict_top1_texts(patched_batch)

    orig_not_target = (orig_top1 != target_index)
    became_target = (patched_top1 == target_index)
    successful = (orig_not_target & became_target)
    ASR = successful.sum() / len(successful) if successful.sum() > 0 else 0

    print("\n===============================")
    print("ATTACK SUMMARY")
    print("===============================\n")
    print(f"Total holdout images: {len(hold_tensors)}")
    print(f"Successful targeted attacks: {successful.sum().item()}")
    print(f"Attack Success Rate (ASR): {ASR*100:.2f}%\n")

    print("Interpretation:")
    print("- ASR measures how often the patch FORCES CLIP to output the target caption.")
    print("- A score of 100% means:")
    print("    Every unseen image was misclassified.")
    print("    Patch GENERALIZED beyond training data.")
    print("    The attack is strong and transferable.\n")

    print("Saving visual comparison examples...")

    nshow = min(8, len(hold_tensors))
    for i in range(nshow):
        orig = (batch[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        pat = (patched_batch[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

        fig,axs = plt.subplots(1,2,figsize=(7,3))
        axs[0].imshow(orig); axs[0].set_title(f"Original:\n{CANDIDATE_TEXTS[orig_top1[i]]}")
        axs[0].axis("off")
        axs[1].imshow(pat); axs[1].set_title(f"Patched:\n{CANDIDATE_TEXTS[patched_top1[i]]}")
        axs[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"example_{i}.png"))
        plt.close(fig)

    print(f"\n Saved example images to: {SAVE_DIR}")
    print("These images show BEFORE/AFTER predictions with patch applied.\n")

print("===============================")
print("ATTACK PIPELINE COMPLETE")
print("===============================")
