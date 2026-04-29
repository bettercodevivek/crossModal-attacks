"""
Carlini–Wagner style L2 adversarial attack (untargeted).
Minimizes c * margin_loss + L2 distance in tanh parameterization (box [0,1]).
"""
import torch
import torch.nn as nn


def cw_l2_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 100,
    lr: float = 0.01,
    c: float = 10.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    """
    Untargeted: minimize c * relu(Z_y - max_{i!=y} Z_i + kappa) + ||x' - x||_2^2.
    When margin <= 0, prediction differs from y.
    """
    orig = x.detach()
    # w -> x_adv = 0.5 * (tanh(w) + 1)
    w = torch.atanh(torch.clamp(orig * 2.0 - 1.0, -1 + 1e-6, 1 - 1e-6)).detach()
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(x_adv)

        z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.unsqueeze(1), False)
        z_max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1).values

        margin = z_y - z_max_other + kappa
        f = torch.clamp(margin, min=0.0).sum()
        l2 = ((x_adv - orig).reshape(x.size(0), -1).pow(2).sum(dim=1)).sum()
        loss = c * f + l2
        loss.backward()
        opt.step()

    x_out = 0.5 * (torch.tanh(w) + 1.0).detach()
    return torch.clamp(x_out, 0.0, 1.0)
