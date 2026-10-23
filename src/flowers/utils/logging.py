"""KATMAN 0 - Altyapı. Amaç: konsol + dosya loglama ve (opsiyonel) W&B/TensorBoard sarmalayıcı."""
import logging, sys
from pathlib import Path


def get_logger(name: str = "flowers", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


class MetricLogger:
    """Tek arayüz: istersen TensorBoard veya wandb aç, yoksa sadece log yazar."""
    def __init__(self, backend: str = "none", run_dir: str = "runs/tmp", project: str = "flowers", name: str = ""):
        self.backend, self.tb, self.wandb = backend, None, None
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(run_dir)
        elif backend == "wandb":
            import wandb
            self.wandb = wandb; wandb.init(project=project, name=name)

    def log(self, metrics: dict, step: int) -> None:
        if self.tb:
            for k, v in metrics.items(): self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)
