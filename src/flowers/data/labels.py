"""KATMAN 1 - Veri. Amaç: ETİKET EŞLEMESİ. Klasör adları '1,10,100' olduğundan ImageFolder alfabetik
indeks üretir (idx != klasör no). Bu dosya idx <-> orijinal sınıf adı eşlemesini kaydeder/yükler."""
import json
from pathlib import Path


def save_class_map(class_to_idx: dict, path: str) -> None:
    idx_to_class = {int(i): c for c, i in class_to_idx.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"idx_to_class": idx_to_class}, indent=1))


def load_class_map(path: str) -> dict[int, str]:
    d = json.loads(Path(path).read_text())["idx_to_class"]
    return {int(k): v for k, v in d.items()}
