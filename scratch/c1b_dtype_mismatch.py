"""C1b - dtype uyumsuzlugu: fp32 agirlikli nn.Linear'a fp16 girdi verilirse ne olur?
Hata mesaji oldugu gibi yazdirilir."""
import torch
import torch.nn as nn

torch.manual_seed(0)

layer = nn.Linear(4, 2)                 # agirliklar fp32
print("agirlik dtype:", layer.weight.dtype)

x32 = torch.randn(1, 4)
print("fp32 girdi ile cikti dtype:", layer(x32).dtype)

x16 = x32.half()
print("girdi dtype:", x16.dtype)
try:
    layer(x16)
    print("HATA VERMEDI")
except Exception as e:
    print("Hata sinifi:", type(e).__name__)
    print("Hata mesaji:", e)

# Cozum: ya girdiyi model dtype'ina cevir, ya modeli girdi dtype'ina cevir
print("\ngirdiyi float() yapinca cikti dtype:", layer(x16.float()).dtype)
