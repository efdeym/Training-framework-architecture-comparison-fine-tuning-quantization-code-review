"""KATMAN 3 - Eğitim. Amaç: tek epoch eğitim ve tahmin döngüleri (saf fonksiyonlar)."""
import torch
from .optim import set_lr


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_steps, warmup_steps,
                    scaler=None, mixer=None, ema=None, teacher=None, distill_fn=None,
                    amp=True, channels_last=True, grad_clip=0.0, step_offset=0):
    model.train()
    run_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        step = step_offset + i
        set_lr(optimizer, step, total_steps, warmup_steps)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if mixer is not None:
            x, y = mixer(x, y)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = distill_fn(loss, out.float(), t_out.float())
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        run_loss += loss.item() * x.size(0); n += x.size(0)
    return run_loss / max(n, 1), step_offset + len(loader)


@torch.inference_mode()
def predict_loader(model, loader, device, channels_last=True):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        x = x.to(device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        logits.append(model(x).float().cpu()); targets.append(y)
    return torch.cat(logits), torch.cat(targets)
