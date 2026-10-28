"""C2 - model.train() ile model.eval() farki (Dropout ve BatchNorm).
Ayni girdi, ayni model, iki farkli mod."""
import torch
import torch.nn as nn

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(8, 8),
    nn.BatchNorm1d(8),
    nn.Dropout(p=0.5),
    nn.Linear(8, 3),
)
bn = model[1]
x = torch.randn(16, 8)  # batch=16 (BatchNorm train modunda batch > 1 ister)

with torch.no_grad():
    # --- Ayni girdi, train modunda iki kez ---
    model.train()
    t1 = model(x)[0]
    t2 = model(x)[0]
    print("[train] 1. cagri, ilk ornek:", t1)
    print("[train] 2. cagri, ilk ornek:", t2)
    print("[train] iki cagri ayni mi? ->", torch.equal(t1, t2), "(Dropout rastgele oldugu icin farkli beklenir)")

    # --- Ayni girdi, eval modunda iki kez ---
    model.eval()
    e1 = model(x)[0]
    e2 = model(x)[0]
    print("\n[eval]  1. cagri, ilk ornek:", e1)
    print("[eval]  2. cagri, ilk ornek:", e2)
    print("[eval]  iki cagri ayni mi?  ->", torch.equal(e1, e2), "(deterministik beklenir)")

    # --- train ile eval ayni girdiye farkli cevap verir ---
    print("\ntrain ciktisi ile eval ciktisi ayni mi? ->", torch.equal(t1, e1))

    # --- BatchNorm running istatistikleri ---
    model.train()
    before = bn.running_mean.clone()
    model(x)
    after_train = bn.running_mean.clone()
    model.eval()
    model(x)
    after_eval = bn.running_mean.clone()
    print("\nBatchNorm running_mean train cagrisi sonrasi degisti mi? ->", not torch.equal(before, after_train))
    print("BatchNorm running_mean eval  cagrisi sonrasi degisti mi? ->", not torch.equal(after_train, after_eval))

print("\nmodel.training bayragi (eval sonrasi):", model.training, "| dropout.training:", model[2].training)
