"""
Local demo UI server: CLIP cross-modal FGSM/PGD, preset gallery, paper CNN eval.

Run from project root:
    python -m uvicorn src.web_server:app --host 127.0.0.1 --port 7860
"""
from __future__ import annotations

import base64
import io
import os
import re
import threading
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from pydantic import BaseModel, Field

from attacks import FGSMAttack, PGDAttack
from config import Config
from demo_attack import load_clip_model
from evaluation.metrics import compute_similarity
from utils import get_device, get_image_preprocessor, tensor_to_image

_REPO_ROOT = _SRC_DIR.parent
_WEBUI = _REPO_ROOT / "webui"
_DATA_HOLDOUT = _REPO_ROOT / "data" / "holdout"
_DATA_IMAGES = _REPO_ROOT / "data" / "images"

_GALLERY_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_clip_lock = threading.Lock()
_model = None
_processor = None
_device: str | None = None

_paper_lock = threading.Lock()


def _ensure_clip():
    global _model, _processor, _device
    with _clip_lock:
        if _model is None:
            Config.ensure_dirs()
            _device = get_device()
            _model, _processor = load_clip_model(_device)
    assert _model is not None and _processor is not None and _device is not None
    return _model, _processor, _device


def _tensor_to_data_url(tensor: torch.Tensor) -> str:
    arr = tensor_to_image(tensor.detach().cpu())
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_resolve_preset(source: str, name: str) -> Path:
    source = (source or "").strip().lower()
    name = os.path.basename((name or "").strip())
    if source not in ("holdout", "images"):
        raise HTTPException(status_code=400, detail="preset_source must be holdout or images")
    if not name or not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid preset file name")
    suf = Path(name).suffix.lower()
    if suf not in _GALLERY_EXT:
        raise HTTPException(status_code=400, detail="preset must be jpg, png, or bmp")
    base = (_DATA_HOLDOUT if source == "holdout" else _DATA_IMAGES).resolve()
    path = (base / name).resolve()
    try:
        path.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid preset path") from e
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {source}/{name}")
    return path


class PaperEvalBody(BaseModel):
    dataset: str
    checkpoint: str = Field(min_length=1)
    attack: str
    limit_batches: int | None = Field(None, ge=1, le=500)
    epsilon: float | None = Field(None, gt=0, le=1)
    pgd_alpha: float | None = Field(None, gt=0, le=1)
    pgd_steps: int | None = Field(None, ge=1, le=500)
    cw_steps: int | None = Field(None, ge=1, le=500)
    cw_lr: float | None = Field(None, gt=0)
    cw_c: float | None = Field(None, gt=0)


def _scan_gallery_dir(folder: Path, source: str, url_prefix: str) -> list[dict]:
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _GALLERY_EXT:
            continue
        out.append(
            {
                "name": p.name,
                "source": source,
                "url": f"{url_prefix}/{p.name}",
            }
        )
    return out


def create_app() -> FastAPI:
    app = FastAPI(title="Cross-Modal Attack Demo", version="1.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if _WEBUI.is_dir():
        app.mount("/static", StaticFiles(directory=str(_WEBUI)), name="static")

    if _DATA_HOLDOUT.is_dir():
        app.mount(
            "/gallery-holdout",
            StaticFiles(directory=str(_DATA_HOLDOUT)),
            name="gallery_holdout",
        )
    if _DATA_IMAGES.is_dir():
        app.mount(
            "/gallery-images",
            StaticFiles(directory=str(_DATA_IMAGES)),
            name="gallery_images",
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "clip_loaded": _model is not None}

    @app.get("/api/gallery")
    def gallery():
        holdout = _scan_gallery_dir(_DATA_HOLDOUT, "holdout", "/gallery-holdout")
        images = _scan_gallery_dir(_DATA_IMAGES, "images", "/gallery-images")
        return {
            "holdout": holdout,
            "images": images,
            "note": (
                "No sample images found under data/holdout or data/images. "
                "Run `python download_images.py` from the project root."
                if not holdout and not images
                else None
            ),
        }

    @app.post("/api/attack")
    def run_attack(
        image: UploadFile | None = File(None),
        attack: str = Form(...),
        target_text: str = Form(...),
        epsilon: float | None = Form(None),
        pgd_steps: int | None = Form(None),
        pgd_alpha: float | None = Form(None),
        preset_name: str | None = Form(None),
        preset_source: str | None = Form(None),
    ):
        target_text = (target_text or "").strip()
        if not target_text:
            raise HTTPException(status_code=400, detail="target_text is required")
        attack = attack.lower().strip()
        if attack not in ("fgsm", "pgd"):
            raise HTTPException(status_code=400, detail="attack must be fgsm or pgd")

        use_preset = bool((preset_name or "").strip())
        raw = None
        if use_preset:
            if image is not None and image.filename:
                raise HTTPException(
                    status_code=400,
                    detail="provide either an uploaded image or a preset, not both",
                )
            path = _safe_resolve_preset(preset_source or "", preset_name or "")
            raw = path.read_bytes()
        else:
            if image is None or not image.filename:
                raise HTTPException(
                    status_code=400,
                    detail="upload an image or choose a sample from the gallery",
                )
            raw = image.file.read()

        if not raw:
            raise HTTPException(status_code=400, detail="empty image")

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

        preprocess = get_image_preprocessor()
        tensor = preprocess(pil)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        model, processor, device = _ensure_clip()
        orig = tensor.to(device)

        eps = (
            float(epsilon)
            if epsilon is not None
            else (Config.FGSM_EPSILON if attack == "fgsm" else Config.PGD_EPSILON)
        )
        steps_out = None
        alpha_out = None

        if attack == "fgsm":
            atk = FGSMAttack(model, processor, target_text, device)
            adv = atk.attack(orig, epsilon=eps)
        else:
            steps = int(pgd_steps) if pgd_steps is not None else Config.PGD_STEPS
            alpha = float(pgd_alpha) if pgd_alpha is not None else Config.PGD_ALPHA
            steps = max(1, min(steps, 200))
            steps_out = steps
            alpha_out = alpha
            atk = PGDAttack(model, processor, target_text, device)
            adv = atk.attack(orig, epsilon=eps, num_steps=steps, alpha=alpha)

        orig_sims = compute_similarity(model, processor, orig, target_text, device)
        adv_sims = compute_similarity(model, processor, adv, target_text, device)

        success = bool(adv_sims[0] > orig_sims[0])
        return {
            "attack": attack.upper(),
            "target_text": target_text,
            "epsilon": eps,
            "pgd_steps": steps_out,
            "pgd_alpha": alpha_out,
            "similarity_original": float(orig_sims[0]),
            "similarity_adversarial": float(adv_sims[0]),
            "confidence_shift": float(adv_sims[0] - orig_sims[0]),
            "attack_succeeded": success,
            "device": device,
            "image_original": _tensor_to_data_url(orig.squeeze(0)),
            "image_adversarial": _tensor_to_data_url(adv.squeeze(0)),
            "preset_used": (
                {"name": preset_name, "source": preset_source}
                if use_preset
                else None
            ),
        }

    @app.get("/api/paper/checkpoints")
    def paper_checkpoints():
        from paper.paper_config import PaperConfig

        PaperConfig.ensure_dirs()
        ckpt_dir = Path(PaperConfig.checkpoint_dir())
        items = []
        if ckpt_dir.is_dir():
            for p in sorted(ckpt_dir.glob("*.pt")):
                items.append(
                    {
                        "filename": p.name,
                        "path": str(p.resolve()),
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    }
                )
        defaults = ["mnist_none.pt", "cifar10_none.pt"]
        return {
            "checkpoint_dir": str(ckpt_dir.resolve()),
            "checkpoints": items,
            "suggested": [f for f in defaults if (ckpt_dir / f).is_file()],
            "note": (
                "No checkpoints found. Train with "
                "`python src/paper/run_paper_experiments.py train ...` "
                "or run `python src/paper/run_paper_experiments.py fast-benchmark`."
                if not items
                else None
            ),
        }

    @app.post("/api/paper/eval")
    def paper_eval(body: PaperEvalBody):
        from paper.data import get_loaders
        from paper.metrics_classify import accuracy, evaluate_robustness
        from paper.models import build_model
        from paper.paper_config import PaperConfig
        from paper.train_models import load_checkpoint

        PaperConfig.ensure_dirs()
        dataset = body.dataset.lower().strip()
        attack = body.attack.lower().strip()
        if dataset not in ("mnist", "cifar10"):
            raise HTTPException(status_code=400, detail="dataset must be mnist or cifar10")
        if attack not in ("fgsm", "pgd", "cw"):
            raise HTTPException(status_code=400, detail="attack must be fgsm, pgd, or cw")
        ckpt_dir = Path(PaperConfig.checkpoint_dir())
        ckpt_name = os.path.basename(body.checkpoint.strip())
        if not ckpt_name or not ckpt_name.endswith(".pt"):
            raise HTTPException(status_code=400, detail="checkpoint must be a .pt filename")
        ckpt_path = (ckpt_dir / ckpt_name).resolve()
        try:
            ckpt_path.relative_to(ckpt_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid checkpoint name") from None
        if not ckpt_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Checkpoint not found: {ckpt_path.name}. Train or run fast-benchmark first.",
            )

        lim = body.limit_batches
        eps = body.epsilon
        if eps is None:
            if attack == "fgsm":
                eps = PaperConfig.FGSM_EPSILON
            elif attack == "pgd":
                eps = PaperConfig.PGD_EPSILON
            else:
                eps = PaperConfig.PGD_EPSILON
        pgd_alpha = body.pgd_alpha if body.pgd_alpha is not None else PaperConfig.PGD_ALPHA
        pgd_steps = body.pgd_steps if body.pgd_steps is not None else PaperConfig.PGD_STEPS
        cw_steps = body.cw_steps if body.cw_steps is not None else PaperConfig.CW_STEPS
        cw_lr = body.cw_lr if body.cw_lr is not None else PaperConfig.CW_LR
        cw_c = body.cw_c if body.cw_c is not None else PaperConfig.CW_C

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _run():
            model = build_model(dataset, device)
            load_checkpoint(model, str(ckpt_path), device)
            _, test_loader = get_loaders(
                dataset, PaperConfig.BATCH_SIZE, num_workers=0
            )
            clean_acc = accuracy(model, test_loader, device)
            metrics = evaluate_robustness(
                model,
                test_loader,
                device,
                attack=attack,
                epsilon=eps,
                pgd_alpha=pgd_alpha,
                pgd_steps=pgd_steps,
                cw_steps=cw_steps,
                cw_lr=cw_lr,
                cw_c=cw_c,
                max_batches=lim,
            )
            return clean_acc, metrics

        with _paper_lock:
            clean_acc, metrics = _run()

        return {
            "dataset": dataset,
            "checkpoint": ckpt_name,
            "attack": attack,
            "device": str(device),
            "clean_accuracy": round(float(clean_acc), 4),
            "evaluated_samples": metrics["total_samples"],
            "successful_attacks": metrics["successful_attacks"],
            "asr_percent": round(float(metrics["asr_percent"]), 4),
            "robustness_score": round(float(metrics["robustness_score"]), 4),
            "limit_batches": lim,
            "epsilon_used": eps,
            "pgd_steps_used": pgd_steps if attack == "pgd" else None,
            "pgd_alpha_used": pgd_alpha if attack == "pgd" else None,
            "cw_steps_used": cw_steps if attack == "cw" else None,
        }

    @app.get("/")
    def index():
        index_path = _WEBUI / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="webui/index.html missing — add the webui folder from the repo.",
            )
        return FileResponse(index_path)

    return app


app = create_app()
