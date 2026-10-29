# import torch
# from torchvision import transforms

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# def build_eval_transform(img_size=224, crop_pct=0.875, mean=IMAGENET_MEAN, std=IMAGENET_STD):
#     resize_size = int(img_size / crop_pct)
#     return transforms.Compose([
#         transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

# def build_train_transform(img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
#     return transforms.Compose([
#         transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

"""KATMAN 1 - Veri. Amaç: eğitim/değerlendirme dönüşümleri. Değerlendirme dönüşümü teslim paketindeki
predict.py ile BİREBİR aynı olmalı (resize -> center crop -> normalize)."""
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(size: int, color_jitter: float = 0.1, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    ops = [
        T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
    ]
    if color_jitter > 0:  # hue YOK: renk çiçek türünü belirler
        ops.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.0))
    ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    return T.Compose(ops)


def build_eval_transform(size: int, crop_pct: float = 0.875, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([
        T.Resize(int(size / crop_pct), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std),
    ])