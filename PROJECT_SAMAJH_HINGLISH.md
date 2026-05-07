# Cross-Modal Attacks — Poora Project (Simple Hinglish)

Yeh repo **adversarial robustness testing** ke liye hai — matlab deep learning models ko **chhoti image perturbations** se test karte hain ki woh kitna **stable / hack-proof** hain. Paper ka naam hai *Cross-Modal Adversarial Attacks: Towards Generalized Robustness Testing in Deep Learning*.

---

## Ek line mein kya hai?

Do alag–alag **tracks** hain:

1. **CLIP cross-modal** — image ko thoda badlo, taaki **text side** (caption similarity) manipulate ho. Vision aur language **ek hi embedding space** mein aate hain; attack yahi shared space ki kamzori dikhata hai.
2. **Paper wale CNN experiments (MNIST / CIFAR-10)** — classic image classification par **FGSM, PGD, CW** jaise attacks + **defenses** (adversarial training, distillation). Yeh paper ke Section V–VI jaisa setup hai.

Dono tracks **independent** hain — tum sirf CLIP chala sakte ho, sirf CNN paper pipeline, ya dono.

---

## Cross-modal attack ka simple matlab

Normal attack: image bigado → **same modality** mein galat label (e.g. cat ko dog).

**Cross-modal (CLIP track):** image bigado → goal hai ki model ki **text embedding** kisi **target sentence** ke zyada paas aa jaye (jaise default: *"a photo of a banana"*). Matlab **vision** se **language behaviour** ko push karna — yahi “cross-modal” idea hai.

---

## Folder structure (short)

| Jagah | Kaam |
|--------|------|
| `src/demo_attack.py` | CLIP attacks run karne ka main entry — patch / FGSM / PGD |
| `src/config.py` | CLIP: patch size, epsilon, steps, target text, paths |
| `src/attacks/` | Patch, FGSM, PGD implementations (CLIP ke liye) |
| `src/evaluation/` | CLIP metrics — similarity, ASR jaise cheezein |
| `src/visualization/` | Results visualize |
| `src/paper/` | MNIST/CIFAR CNN train + classify attacks + plots + CLI |
| `download_images.py` | CLIP demo ke liye Unsplash se images download |
| `src/web_server.py` | Optional **local web UI** (FastAPI) — CLIP FGSM/PGD demo + thoda paper side |
| `PRESENTATION.md` | Presentation-style notes (agar repo mein hai) |

Outputs generally: `src/results/` (CLIP), `paper_checkpoints/`, `paper_results/`, kabhi `cnn_paper_benchmark/` (fast benchmark).

---

## Track 1: CLIP attacks (`demo_attack.py`)

**Kya hota hai:** Pretrained **CLIP** load hota hai. Tum images par attack lagate ho taaki unka **cosine similarity** target caption ke saath badhe.

**Teen attack types (high level):**

- **Universal patch** — ek fixed patch train karte ho jo bahut images par kaam kare (transfer).
- **FGSM** — ek step gradient; tez, thoda weak ho sakta hai.
- **PGD** — kai steps; zyada strong adversarial examples.

**Default flow:** Training images se patch train (agar patch), holdout par evaluate; comparisons + `metrics.json`.

**Pehle:** `python download_images.py` se `data/images` aur `data/holdout` bhar lo (CLIP track ke liye).

---

## Track 2: Paper CNN pipeline (`src/paper/`)

**Kya hota hai:** MNIST ya CIFAR-10 par **CNN** train → phir **white-box** attacks:

- **FGSM** — fast one-step  
- **PGD** — iterative, zyada reliable attack metric  
- **CW-style L2** — optimization based, slow par strong  

**Metrics:** **ASR** (attack success rate) — kitni samples attack se fool ho gayi; **R = 1 − ASR** robustness jaisa interpret karte ho.

**Defenses:**

- Normal training  
- **Adversarial training** — training mein hi adversarial examples mix  
- **Defensive distillation** — teacher se soft labels copy  

CLI example README mein detail hai: `train`, `eval`, `reproduce-table`, `fast-benchmark`.

---

## Dependencies (rough idea)

`torch`, `torchvision`, `transformers` (CLIP), numpy, matplotlib, tqdm, requests; web UI ke liye **FastAPI + uvicorn**. Sab `requirements.txt` mein listed hai.

---

## Kyun useful hai?

- **Research / padhai:** multimodal models ki kamzorian samajhna  
- **Robustness:** model ko quantitatively test karna attack ke against  
- **Paper reproduction:** CNN side ka methodology aligned hai Sections V–VI ke saath  

---

## Quick mental map

```
[Tumhari image]  →  [Attack: FGSM / PGD / Patch]  →  [CLIP ya CNN]
                              ↓
                    [Metrics: ASR, similarity shift, robustness R]
```

Agar sirf **simple demo** chahiye: CLIP track `demo_attack.py` se start karo. Agar **tables / CNN paper numbers** chahiye: `run_paper_experiments.py` use karo project root se.

---

*Yeh document poora project ka bird’s-eye view hai; commands aur exact paths ke liye `README.md` authoritative hai.*
