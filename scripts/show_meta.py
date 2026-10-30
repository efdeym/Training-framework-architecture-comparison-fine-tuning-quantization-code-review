import sys, json, torch
for p in sys.argv[1:]:
    extra = {"meta.json": ""}
    torch.jit.load(p, map_location="cpu", _extra_files=extra)
    m = json.loads(extra["meta.json"])
    m.pop("idx_to_class", None)
    print(p, m)