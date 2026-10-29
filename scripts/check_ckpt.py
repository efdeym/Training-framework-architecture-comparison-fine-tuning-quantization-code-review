import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
need = ["model", "arch", "class_to_idx", "data_cfg", "img_size"]
missing = [k for k in need if k not in ck]
print("anahtarlar:", list(ck.keys()))
print("eksik:", missing or "yok")
print("arch:", ck["arch"], "| img_size:", ck["img_size"], "| sinif:", len(ck["class_to_idx"]),
      "| crop_pct:", ck.get("crop_pct"), "| epoch:", ck.get("epoch"))
print("ilk 5 esleme:", list(ck["class_to_idx"].items())[:5])
sys.exit(1 if missing else 0)