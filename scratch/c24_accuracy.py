"""C24 - valid kumesinde top-1: fp32 ve fp16-yuvarlanmis agirliklar (CPU).

Notlar:
- Degerlendirme yolu scripts/evaluate.py ile aynidir (ImageFolder + build_eval_transform + predict_loader).
- 'fp16-yuvarlanmis': agirliklar fp16'ya cevrilip fp32'ye geri alinir. CPU'da teslim modelinin
  (fp16 TorchScript, benchmark.py/predict.py CPU'da .float() yapar) hesaplamasiyla ayni etkiyi verir.
- Statik/dinamik INT8 icin dogruluk OLCULMEZ (dinamik donusmedi; statikte gercek kalibrasyon yok).
- Dosya yazmaz; yalnizca terminale basar.
Kullanim: python scratch/c24_accuracy.py <valid_klasoru> <ckpt1> [<ckpt2> ...]
"""
import copy
import sys
import warnings

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore")
from flowers.data.transforms import build_eval_transform
from flowers.evaluation.metrics import top1_accuracy
from flowers.export.packager import load_checkpoint_model
from flowers.training.engine import predict_loader

if __name__ == "__main__":
    data, ckpts = sys.argv[1], sys.argv[2:]
    dev = torch.device("cpu")
    for path in ckpts:
        model, ck = load_checkpoint_model(path)
        tf = build_eval_transform(ck["img_size"], ck["crop_pct"], ck["data_cfg"]["mean"], ck["data_cfg"]["std"])
        ds = ImageFolder(data, transform=tf)
        remap = torch.tensor([ck["class_to_idx"][c] for c in ds.classes])
        loader = DataLoader(ds, batch_size=64, num_workers=0)

        def acc(m):
            logits, targets = predict_loader(m.to(dev), loader, dev, channels_last=False)
            return top1_accuracy(logits, remap[targets]), logits

        a32, l32 = acc(model)
        m16 = copy.deepcopy(model).half().float()
        a16, l16 = acc(m16)
        agree = (l32.argmax(1) == l16.argmax(1)).float().mean().item()
        print(f"{path}: n={len(ds)} top1 fp32={a32:.4f} fp16-yuvarlanmis={a16:.4f} tahmin_uyumu={agree:.4f}")

