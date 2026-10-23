"""KATMAN 5 - Sıkıştırma. Amaç: model boyutunu küçültme.
  1) to_fp16         : ağırlıkları fp16 yap (2x küçük, neredeyse kayıpsız; GPU için ideal)
  2) static_ptq_int8 : FX graph mode statik int8 (CPU; CNN'lerde ~4x küçük, hız kazancı olabilir).
Not: int8 yolu modele göre sorun çıkarabilir (hardswish, SE vb.). Doğruluğu mutlaka tekrar ölç."""
import copy
import torch


def to_fp16(model):
    return copy.deepcopy(model).half()


def static_ptq_int8(model, calib_loader, backend: str = "x86", n_batches: int = 20):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    torch.backends.quantized.engine = backend
    m = copy.deepcopy(model).cpu().eval()
    example = next(iter(calib_loader))[0][:1]
    prepared = prepare_fx(m, get_default_qconfig_mapping(backend), example_inputs=(example,))
    with torch.inference_mode():
        for i, (x, _) in enumerate(calib_loader):
            prepared(x)
            if i + 1 >= n_batches:
                break
    return convert_fx(prepared)
