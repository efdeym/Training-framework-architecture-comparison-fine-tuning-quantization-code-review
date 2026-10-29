"""KATMAN 1 - Veri. Amaç: klasör yapısından (train/valid/test) Dataset ve DataLoader üretmek,
dengesizlik için sampler, final eğitim için split birleştirme."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from .transforms import build_train_transform, build_eval_transform


def _check_same_classes(*sets):
    ref = sets[0].class_to_idx
    for s in sets[1:]:
        assert s.class_to_idx == ref, "Split'ler arasında sınıf klasörleri farklı!"

def drop_files(ds: ImageFolder, names) -> ImageFolder:
    """Dosya adı `names` listesinde olan örnekleri çıkarır (örn. train-valid mükerrer görsel)."""
    names = set(names or [])
    if not names:
        return ds
    keep = [i for i, (p, _) in enumerate(ds.samples) if Path(p).name not in names]
    ds.samples = [ds.samples[i] for i in keep]
    ds.imgs = ds.samples
    ds.targets = [ds.targets[i] for i in keep]
    return ds

def build_datasets(cfg, mean, std):
    root = Path(cfg.data.root)
    tr_t = build_train_transform(cfg.data.img_size, cfg.aug.color_jitter, mean, std)
    ev_t = build_eval_transform(cfg.data.test_img_size, cfg.data.crop_pct, mean, std)
    train = drop_files(ImageFolder(root / "train", tr_t), cfg.data.exclude_train)
    valid = ImageFolder(root / "valid", ev_t)
    test = ImageFolder(root / "test", ev_t) if (root / "test").exists() else None
    _check_same_classes(*[s for s in (train, valid, test) if s is not None])
    return train, valid, test


def make_weights(dataset: ImageFolder) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(dataset.targets))
    w = 1.0 / counts.float().sqrt()          # karekök: hafif dengeleme (tam 1/n aşırıdır)
    return w[torch.tensor(dataset.targets)]


def build_loaders(cfg, mean, std):
    train, valid, test = build_datasets(cfg, mean, std)
    class_to_idx = train.class_to_idx
    train_ds = train
    if cfg.data.final_fit:               # SON eğitim: train + valid (+ test). Model seçimi bittikten sonra!
        tr_t = build_train_transform(cfg.data.img_size, cfg.aug.color_jitter, mean, std)
        extra = [ImageFolder(Path(cfg.data.root) / "valid", tr_t)]
        if cfg.data.include_test_in_final and test is not None:
            extra.append(ImageFolder(Path(cfg.data.root) / "test", tr_t))
        train_ds = ConcatDataset([train] + extra)
    sampler = None
    if cfg.data.balanced_sampler and not cfg.data.final_fit:
        w = make_weights(train)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    kw = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    valid_loader = DataLoader(valid, shuffle=False, **kw)
    test_loader = DataLoader(test, shuffle=False, **kw) if test is not None else None
    return train_loader, valid_loader, test_loader, class_to_idx
