"""KATMAN 3 - Eğitim. Amaç: kayıp fonksiyonları (label smoothing, MixUp soft-label)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCE(nn.Module):
    """MixUp/CutMix sonrası hedefler [B, C] olasılık vektörüdür."""
    def forward(self, logits, target):
        return torch.sum(-target * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()


def build_criterion(label_smoothing: float, use_mix: bool):
    return SoftTargetCE() if use_mix else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
