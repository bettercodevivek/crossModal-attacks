"""Standard training, adversarial training, and defensive distillation."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .classification_attacks import pgd_attack
from .paper_config import PaperConfig


def train_epoch_standard(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def train_epoch_adversarial(model, loader, optimizer, device, cfg: PaperConfig):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = pgd_attack(
            model,
            x,
            y,
            cfg.ADV_TRAIN_EPS,
            cfg.ADV_TRAIN_ALPHA,
            cfg.ADV_TRAIN_STEPS,
            random_start=True,
        )
        model.train()
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def train_epoch_distillation(student, teacher, loader, optimizer, device, cfg: PaperConfig):
    student.train()
    teacher.eval()
    T = cfg.DISTILL_TEMPERATURE
    a = cfg.DISTILL_ALPHA
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student(x)
        soft = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        hard = F.cross_entropy(s_logits, y)
        loss = a * soft + (1.0 - a) * hard
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    lr: float,
    weight_decay: float,
    defense: str,
    teacher: nn.Module,
    cfg: PaperConfig,
):
    """
    defense: 'none' | 'adversarial' | 'distillation'
    teacher: required if defense == 'distillation'
    """
    defense = defense.lower()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        if defense == "none":
            loss = train_epoch_standard(model, train_loader, opt, device)
        elif defense == "adversarial":
            loss = train_epoch_adversarial(model, train_loader, opt, device, cfg)
        elif defense == "distillation":
            if teacher is None:
                raise ValueError("Teacher required for distillation")
            loss = train_epoch_distillation(
                model, teacher, train_loader, opt, device, cfg
            )
        else:
            raise ValueError(defense)

        acc = eval_accuracy(model, test_loader, device)
        print(f"  Epoch {epoch}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}")

    return model


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_checkpoint(model, path: str, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model
