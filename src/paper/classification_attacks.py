"""
White-box L∞ attacks on classification models (FGSM, PGD).
Inputs x are in [0, 1]; gradients flow through NormalizedModel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _clamp(x: torch.Tensor, orig: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.clamp(x, orig - epsilon, orig + epsilon)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Untargeted FGSM: maximize CE loss to flip prediction.
    x_adv = x + epsilon * sign(grad_x CE(f(x), y))
    """
    x = x.detach().clone()
    x.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + epsilon * grad.sign()
    x_adv = x_adv.detach()
    return torch.clamp(x_adv, 0.0, 1.0)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    """Untargeted PGD (L∞)."""
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = _clamp(x_adv, x, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def predict_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    pred = model(x).argmax(dim=1)
    correct = pred.eq(y).float()
    return pred, correct


def batch_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    attack_name: str,
    epsilon: float,
    pgd_alpha: float,
    pgd_steps: int,
) -> torch.Tensor:
    """Run attack on a batch; attack_name in {'fgsm','pgd'}."""
    if attack_name == "fgsm":
        return fgsm_attack(model, x, y, epsilon)
    if attack_name == "pgd":
        return pgd_attack(model, x, y, epsilon, pgd_alpha, pgd_steps)
    raise ValueError(attack_name)
