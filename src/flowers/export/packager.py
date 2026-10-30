# """KATMAN 6 - Teslim. Amaç: eğitilmiş modeli TEK DOSYA (TorchScript) olarak paketlemek.
# Değerlendiren kişi timm kurmak zorunda kalmaz. Sınıf eşlemesi ve ön işleme ayarları dosyanın içine gömülür."""
# import json
# import torch
# from ..models.factory import create_model


# def load_checkpoint_model(ckpt_path: str):
#     ck = torch.load(ckpt_path, map_location="cpu")
#     model = create_model(ck["arch"], ck["num_classes"], pretrained=False)
#     model.load_state_dict(ck["model"])
#     return model.eval(), ck


# def export_torchscript(model, ck: dict, out_path: str, dtype: str = "fp32"):
#     size = ck["img_size"]
#     meta = {
#         "idx_to_class": {int(i): c for c, i in ck["class_to_idx"].items()},
#         "mean": list(ck["data_cfg"]["mean"]), "std": list(ck["data_cfg"]["std"]),
#         "img_size": size, "crop_pct": ck["crop_pct"], "dtype": dtype, "arch": ck["arch"],
#     }
#     model = model.eval().float()
#     traced = torch.jit.trace(model, torch.randn(1, 3, size, size), check_trace=False)
#     if dtype == "fp32":
#         traced = torch.jit.freeze(traced)
#     elif dtype == "fp16":
#         traced = traced.half()
#     torch.jit.save(traced, out_path, _extra_files={"meta.json": json.dumps(meta)})
#     return meta

"""KATMAN 6 - Teslim. Amaç: eğitilmiş modeli TEK DOSYA (TorchScript) olarak paketlemek.
Değerlendiren kişi timm kurmak zorunda kalmaz. Sınıf eşlemesi ve ön işleme ayarları dosyanın içine gömülür.
Normalizasyon (mean/std) modelin İÇİNDE: model 0-1 aralığında RGB tensör (N,3,H,W) alır."""
import json
import torch
from ..models.factory import create_model


class _NormWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        # Girdi ham [0,1] ise normalize et; zaten normalize (negatif değer içeren) geldiyse dokunma.
        ham = (x.amin(dim=(1, 2, 3), keepdim=True) >= 0).to(x.dtype)
        x = ham * ((x - self.mean) / self.std) + (1 - ham) * x
        return self.model(x)


def load_checkpoint_model(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = create_model(ck["arch"], ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def export_torchscript(model, ck: dict, out_path: str, dtype: str = "fp32"):
    size = ck["img_size"]
    mean, std = list(ck["data_cfg"]["mean"]), list(ck["data_cfg"]["std"])
    meta = {
        "idx_to_class": {int(i): c for c, i in ck["class_to_idx"].items()},
        "mean": mean, "std": std,
        "img_size": size, "crop_pct": ck["crop_pct"], "dtype": dtype, "arch": ck["arch"],
        "norm_in_model": True,
    }
    wrapped = _NormWrapper(model.eval().float(), mean, std).eval()
    traced = torch.jit.trace(wrapped, torch.rand(1, 3, size, size), check_trace=False)
    if dtype == "fp32":
        traced = torch.jit.freeze(traced)
    elif dtype == "fp16":
        traced = traced.half()
    torch.jit.save(traced, out_path, _extra_files={"meta.json": json.dumps(meta)})
    return meta