"""KATMAN 0 - Konfigürasyon. Amaç: tüm deney ayarları tek yerde (YAML -> dataclass).
Herkes aynı şemayı kullanır; deney = bir YAML dosyası. Komut satırından `a.b=c` ile override edilir."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class DataCfg:
    root: str = "data/flowers"          # içinde train/ valid/ test/ olmalı
    img_size: int = 224                 # eğitim kırpma boyutu
    test_img_size: int = 224            # değerlendirme boyutu (FixRes: genelde >= img_size)
    crop_pct: float = 0.875
    batch_size: int = 64
    num_workers: int = 4
    balanced_sampler: bool = False      # sınıf dengesizliği için hafif sampler
    final_fit: bool = False             # True: train+valid(+test) birleşik eğitim (SON adım!)
    include_test_in_final: bool = True


@dataclass
class AugCfg:
    color_jitter: float = 0.1           # DİKKAT: hue=0 sabit. Çiçekte renk = etiket.
    mixup: float = 0.0
    cutmix: float = 0.0
    label_smoothing: float = 0.1


@dataclass
class ModelCfg:
    name: str = "mobilenetv3_large_100"
    pretrained: bool = True
    drop_rate: float = 0.2
    drop_path_rate: float = 0.0


@dataclass
class TrainCfg:
    epochs: int = 30
    lr: float = 1e-3
    head_lr_mult: float = 10.0          # sınıflandırıcı kafa daha yüksek LR
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    linear_probe_epochs: int = 3        # LP-FT: önce sadece kafa
    amp: bool = True
    ema_decay: float = 0.0              # 0 = kapalı, ör. 0.999
    channels_last: bool = True
    grad_clip: float = 1.0


@dataclass
class DistillCfg:
    enabled: bool = False
    teacher_name: str = "convnext_base.fb_in22k_ft_in1k"
    teacher_ckpt: str = ""              # öğretmen önce flowers üzerinde fine-tune edilmiş olmalı
    alpha: float = 0.5
    temperature: float = 4.0


@dataclass
class Config:
    exp_name: str = "baseline"
    seed: int = 42
    out_dir: str = "runs"
    logger: str = "none"                # none | tensorboard | wandb
    data: DataCfg = field(default_factory=DataCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    distill: DistillCfg = field(default_factory=DistillCfg)

    def to_dict(self):
        return asdict(self)


def _merge(dc, d: dict):
    for k, v in (d or {}).items():
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__"):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    base = Path(path).parent / "base.yaml"       # her config önce base.yaml'ı uygular, sonra kendi farklarını
    if base.exists() and Path(path).name != "base.yaml":
        _merge(cfg, yaml.safe_load(base.read_text()))
    with open(path) as f:
        _merge(cfg, yaml.safe_load(f))
    for ov in overrides or []:             # örn: train.epochs=5
        key, val = ov.split("=", 1)
        *parents, last = key.split(".")
        obj = cfg
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, last, yaml.safe_load(val))
    return cfg
