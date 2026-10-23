"""KATMAN 4 - Değerlendirme. Amaç: top-1, sınıf bazlı doğruluk, karışıklık matrisi."""
import torch


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


def per_class_accuracy(logits, targets, num_classes):
    pred = logits.argmax(1)
    acc = torch.zeros(num_classes)
    for c in range(num_classes):
        m = targets == c
        acc[c] = (pred[m] == c).float().mean() if m.any() else float("nan")
    return acc


def confusion_matrix(logits, targets, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets, logits.argmax(1)):
        cm[t, p] += 1
    return cm
