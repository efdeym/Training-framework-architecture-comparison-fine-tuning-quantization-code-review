"""KATMAN 3 - Eğitim. Amaç: AdamW (backbone/kafa ayrı LR) + warmup'lı cosine scheduler (adım bazlı)."""
import math
import torch
from ..models.factory import split_params


def build_optimizer(model, lr: float, head_lr_mult: float, weight_decay: float):
    backbone, head = split_params(model)
    groups = [
        {"params": backbone, "lr": lr, "base_lr": lr},
        {"params": head, "lr": lr * head_lr_mult, "base_lr": lr * head_lr_mult},
    ]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def set_lr(optimizer, step: int, total: int, warmup: int):
    f = cosine_with_warmup(step, total, warmup)
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * f
