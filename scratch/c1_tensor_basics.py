"""C1 - PyTorch temelleri: tensor, shape, dtype, device."""
import torch

print("torch:", torch.__version__)
print("cuda var mi:", torch.cuda.is_available())

# 1) Tensor olusturma
x = torch.arange(12)
print("\n[1] arange(12)        ->", x, "| shape:", tuple(x.shape), "| dtype:", x.dtype)

# 2) Shape degistirme (veri ayni, bakis acisi degisir)
m = x.reshape(3, 4)
print("[2] reshape(3, 4)     -> shape:", tuple(m.shape))

# 3) Gorsel formati: [batch, kanal, yukseklik, genislik]
img = torch.rand(1, 3, 224, 224)
print("[3] gorsel batch      -> shape:", tuple(img.shape), "| dtype:", img.dtype)

# 4) dtype: tamsayidan float'a, float32'den float16'ya
f32 = x.float()
f16 = f32.half()
print("[4] dtype             ->", x.dtype, "->", f32.dtype, "->", f16.dtype)
print("    eleman basina bayt: int64 =", x.element_size(), "| fp32 =", f32.element_size(), "| fp16 =", f16.element_size())

# 5) device: tensor hangi cihazda
print("[5] device            ->", img.device)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("    secilen cihaz     ->", dev)
print("    img.to(dev).device ->", img.to(dev).device)

# 6) Islem icin dtype ve device uyumu gerekir
a = torch.ones(2, 2)
b = torch.ones(2, 2, dtype=torch.float64)
print("[6] float32 + float64 ->", (a + b).dtype, "(PyTorch tipi otomatik yukseltti)")
try:
    torch.ones(2, 3) @ torch.ones(2, 3)
except RuntimeError as e:
    print("    yanlis shape hatasi:", str(e).splitlines()[0])

# 7) Normalize: (x - mean) / std, broadcasting ile
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
norm = (img - mean) / std
print("[7] normalize         -> mean shape:", tuple(mean.shape), "| sonuc shape:", tuple(norm.shape))
