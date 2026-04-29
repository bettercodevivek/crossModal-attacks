"""Classification robustness metrics from the paper: ASR, R = 1 - ASR."""
import torch
import torch.nn as nn
from tqdm import tqdm

from .classification_attacks import batch_attack
from .cw_attack import cw_l2_attack


@torch.no_grad()
def accuracy(model: nn.Module, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def accuracy_max_batches(model: nn.Module, loader, device, max_batches: int):
    """Same metric as `accuracy` but only over the first `max_batches` batches."""
    model.eval()
    correct = 0
    total = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def evaluate_robustness(
    model: nn.Module,
    loader,
    device: torch.device,
    attack: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
    cw_steps: int,
    cw_lr: float,
    cw_c: float,
    max_batches: int = None,
) -> dict:
    """
    ASR: proportion of test samples misclassified after attack (paper Section 5.3, Table 2).
    R = 1 - ASR (robustness score).
    """
    model.eval()
    n_total = 0
    n_adv_success = 0  # misclassified after attack

    for bi, (x, y) in enumerate(tqdm(loader, desc=f"Attack {attack}")):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        if attack == "cw":
            x_adv = cw_l2_attack(
                model, x, y, steps=cw_steps, lr=cw_lr, c=cw_c
            )
        else:
            x_adv = batch_attack(
                model, x, y, attack, epsilon, pgd_alpha, pgd_steps
            )

        with torch.no_grad():
            pred_adv = model(x_adv).argmax(dim=1)

        n_total += y.size(0)
        n_adv_success += pred_adv.ne(y).sum().item()

    asr = n_adv_success / max(n_total, 1)
    return {
        "total_samples": n_total,
        "successful_attacks": n_adv_success,
        "asr": asr,
        "asr_percent": asr * 100.0,
        "robustness_score": 1.0 - asr,
    }
