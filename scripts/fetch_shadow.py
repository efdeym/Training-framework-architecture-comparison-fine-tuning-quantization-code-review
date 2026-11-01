import csv, json, os, time, urllib.parse, urllib.request
from PIL import Image

OUT = r"data\shadow_test"
PER_CLASS = 8
UA = "flower-classifier-student-test/1.0 (university course project)"

# sinif numarasi (klasor adi): arama terimi
CLASSES = {
    6: "Lilium lancifolium", 8: "Strelitzia reginae", 11: "Antirrhinum majus",
    15: "Iris pseudacorus", 17: "Echinacea purpurea", 25: "Muscari",
    26: "Papaver rhoeas", 29: "artichoke flower Cynara", 31: "Dianthus caryophyllus",
    42: "Narcissus pseudonarcissus", 48: "Ranunculus acris", 49: "Leucanthemum vulgare",
    50: "Taraxacum officinale flower", 51: "Petunia hybrida", 54: "Helianthus annuus",
    67: "Crocus vernus", 72: "azalea Rhododendron flowers", 73: "Nymphaea water lily",
    74: "Rosa rose flower", 76: "Ipomoea morning glory", 77: "Passiflora incarnata",
    78: "Nelumbo nucifera", 80: "Anthurium andraeanum", 81: "Plumeria frangipani",
    82: "Clematis", 83: "Hibiscus rosa-sinensis", 84: "Aquilegia",
    87: "Magnolia grandiflora flower", 88: "Cyclamen persicum", 94: "Digitalis purpurea",
    95: "Bougainvillea", 96: "Camellia japonica",
}


def api(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


os.makedirs(OUT, exist_ok=True)
rows = []
for cls, term in CLASSES.items():
    folder = os.path.join(OUT, str(cls))
    os.makedirs(folder, exist_ok=True)
    try:
        data = api({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{term} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 40,
            "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 640,
        })
        pages = sorted(data.get("query", {}).get("pages", {}).values(),
                       key=lambda p: p.get("index", 0))
    except Exception as e:
        print(f"sinif {cls}: arama hatasi: {e}")
        continue
    got = 0
    for p in pages:
        if got >= PER_CLASS:
            break
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") != "image/jpeg" or not info.get("thumburl"):
            continue
        path = os.path.join(folder, f"{got + 1:02d}.jpg")
        try:
            download(info["thumburl"], path)
            with Image.open(path) as im:
                if min(im.size) < 256:
                    os.remove(path)
                    continue
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            print(f"  atlandi: {e}")
            continue
        got += 1
        rows.append([cls, term, p.get("title", ""), info.get("descriptionurl", "")])
        time.sleep(0.7)
    print(f"sinif {cls} ({term}): {got} gorsel")

with open(os.path.join(OUT, "kaynaklar.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["sinif", "arama", "dosya", "kaynak"]] + rows)
print("bitti:", len(rows), "gorsel ->", OUT)