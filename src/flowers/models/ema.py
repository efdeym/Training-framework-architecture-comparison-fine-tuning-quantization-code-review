"""KATMAN 2 - Model. Amaç: Exponential Moving Average ağırlıklar (çoğu zaman bedava doğruluk)."""
import copy
import torch


class ModelEma:
    def __init__(self, model, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
