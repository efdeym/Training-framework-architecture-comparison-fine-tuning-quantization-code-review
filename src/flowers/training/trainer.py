"""KATMAN 3 - Eğitim. Amaç: tüm parçaları birleştiren Trainer: LP-FT, AMP, EMA, MixUp/CutMix, distillation,
checkpoint (en iyi + son). Checkpoint içine class_to_idx ve veri config'i de yazılır."""
from pathlib import Path
import torch
from torchvision.transforms import v2 as T

from ..models.factory import create_model, get_data_config, split_params, count_params
from ..models.ema import ModelEma
from ..data.datasets import build_loaders
from ..data.labels import save_class_map
from ..evaluation.metrics import top1_accuracy
from ..compression.distill import load_teacher, make_distill_fn
from ..utils.logging import get_logger, MetricLogger
from .losses import build_criterion
from .optim import build_optimizer
from .engine import train_one_epoch, predict_loader


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(cfg.out_dir) / cfg.exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("flowers", str(self.run_dir / "train.log"))
        self.mlog = MetricLogger(cfg.logger, str(self.run_dir), name=cfg.exp_name)

        # normalizasyon, modelin pretrained config'inden gelir
        probe = create_model(cfg.model.name, 2, pretrained=False)
        self.data_cfg = get_data_config(probe); del probe
        (self.train_loader, self.valid_loader, self.test_loader, self.class_to_idx) = build_loaders(
            cfg, self.data_cfg["mean"], self.data_cfg["std"])
        self.num_classes = len(self.class_to_idx)
        save_class_map(self.class_to_idx, str(self.run_dir / "class_map.json"))

        self.model = create_model(cfg.model.name, self.num_classes, cfg.model.pretrained,
                                  cfg.model.drop_rate, cfg.model.drop_path_rate).to(self.device)
        if cfg.train.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.log.info(f"{cfg.model.name}: {count_params(self.model):.2f}M param, device={self.device}")

        self.use_mix = cfg.aug.mixup > 0 or cfg.aug.cutmix > 0
        self.criterion = build_criterion(cfg.aug.label_smoothing, self.use_mix)
        self.mixer = self._build_mixer()
        self.optimizer = build_optimizer(self.model, cfg.train.lr, cfg.train.head_lr_mult, cfg.train.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and self.device.type == "cuda")
        self.ema = ModelEma(self.model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
        self.teacher, self.distill_fn = None, None
        if cfg.distill.enabled:
            self.teacher = load_teacher(cfg.distill.teacher_name, cfg.distill.teacher_ckpt, self.num_classes, self.device)
            self.distill_fn = make_distill_fn(cfg.distill.alpha, cfg.distill.temperature)

    def _build_mixer(self):
        a = self.cfg.aug
        if not self.use_mix:
            return None
        ops = []
        if a.mixup > 0: ops.append(T.MixUp(alpha=a.mixup, num_classes=self.num_classes))
        if a.cutmix > 0: ops.append(T.CutMix(alpha=a.cutmix, num_classes=self.num_classes))
        return T.RandomChoice(ops)

    def _set_backbone_trainable(self, flag: bool):
        backbone, _ = split_params(self.model)
        for p in backbone:
            p.requires_grad_(flag)

    def _eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def evaluate(self, loader):
        logits, targets = predict_loader(self._eval_model(), loader, self.device, self.cfg.train.channels_last)
        return top1_accuracy(logits, targets)

    def fit(self):
        c = self.cfg.train
        steps_per_epoch = len(self.train_loader)
        total, warm = c.epochs * steps_per_epoch, c.warmup_epochs * steps_per_epoch
        step, best, val = 0, -1.0, 0.0
        for epoch in range(c.epochs):
            lp = epoch < c.linear_probe_epochs        # LP-FT: ilk epoch'larda sadece kafa
            self._set_backbone_trainable(not lp)
            loss, step = train_one_epoch(
                self.model, self.train_loader, self.criterion, self.optimizer, self.device, epoch, total, warm,
                self.scaler, self.mixer, self.ema, self.teacher, self.distill_fn,
                c.amp, c.channels_last, c.grad_clip, step_offset=step)
            if self.cfg.data.final_fit:
                # valid eğitim verisinin parçası: burada ölçülen skor bağımsız değildir.
                self.log.info(f"epoch {epoch+1}/{c.epochs} loss={loss:.4f} [FINAL FIT]"
                              f"{' [LP]' if lp else ''}")
                self.mlog.log({"train/loss": loss}, epoch)
                continue
            val = self.evaluate(self.valid_loader)
            self.log.info(f"epoch {epoch+1}/{c.epochs} loss={loss:.4f} val_top1={val:.4f}{' [LP]' if lp else ''}")
            self.mlog.log({"train/loss": loss, "val/top1": val}, epoch)
            if val > best:
                best = val
                self._save("best.pt", epoch, val)
        # Test seti her deney sonunda otomatik değerlendirilmez.
        # Model seçimi tamamlanınca scripts/evaluate.py ile yalnızca bir kez değerlendirin.
        self._save("last.pt", c.epochs - 1, None if self.cfg.data.final_fit else val)
        return None if self.cfg.data.final_fit else best

    def _save(self, name, epoch, val):
        m = self._eval_model()
        torch.save({"model": m.state_dict(), "arch": self.cfg.model.name, "num_classes": self.num_classes,
                    "class_to_idx": self.class_to_idx, "data_cfg": self.data_cfg,
                    "img_size": self.cfg.data.test_img_size, "crop_pct": self.cfg.data.crop_pct,
                    "epoch": epoch, "val_top1": val, "config": self.cfg.to_dict()}, self.run_dir / name)
