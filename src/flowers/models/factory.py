"""KATMAN 2 - Model. Amaç: isimden model üretmek (timm). Mimari karşılaştırması sadece config'deki
`model.name` değiştirilerek yapılır. Modelin mean/std/çözünürlük bilgisini de döndürür."""
import timm
import torch.nn as nn


def create_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0) -> nn.Module:
    kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate > 0:
        kw["drop_path_rate"] = drop_path_rate
    return timm.create_model(name, **kw)


def get_data_config(model: nn.Module) -> dict:
    """Modelin önceden eğitildiği mean/std/input_size. Normalizasyon bunlarla yapılmalı."""
    cfg = timm.data.resolve_data_config({}, model=model)
    return {"mean": tuple(cfg["mean"]), "std": tuple(cfg["std"]), "input_size": cfg["input_size"]}


def split_params(model: nn.Module):
    """(backbone_params, head_params). Kafa için ayrı LR ve LP-FT için kullanılır."""
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, list(head.parameters())


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
