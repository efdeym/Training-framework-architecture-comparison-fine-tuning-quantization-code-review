import os, sys

def files(root):
    out = set()
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.add((f, os.path.getsize(os.path.join(dp, f))))
    return out

mine = files(r"data\flowers")
new = files(sys.argv[1])
common = mine & new
print(f"indirilen: {len(new)} gorsel | data\\flowers ile ayni (ad + boyut): {len(common)}")
print("YENI veri:", len(new - mine), "gorsel")