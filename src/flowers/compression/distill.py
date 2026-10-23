"""KATMAN 5 - Sıkıştırma. Amaç: Knowledge Distillation. Büyük öğretmen -> küçük öğrenci.
Öğretmen önce çiçek verisinde fine-tune edilmiş olmalı (teacher_ckpt)."""
import torch
import torch.nn.functional as F
from ..models.factory import create_model


def load_teacher(name: str, ckpt: str, num_classes: int, device):
    t = create_model(name, num_classes, pretrained=not ckpt)
    if ckpt:
        t.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return t.to(device).eval()


def make_distill_fn(alpha: float, T: float):
    def fn(ce_loss, student_logits, teacher_logits):
        kd = F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1),
                      reduction="batchmean") * (T * T)
        return (1 - alpha) * ce_loss + alpha * kd
    return fn
