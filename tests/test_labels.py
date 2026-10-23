"""Etiket eşlemesi: idx != klasör numarası hatasını yakalar."""
from flowers.data.labels import save_class_map, load_class_map


def test_roundtrip(tmp_path):
    c2i = {"1": 0, "10": 1, "100": 2, "2": 3}     # alfabetik sıralama: '2' sonda!
    save_class_map(c2i, str(tmp_path / "m.json"))
    m = load_class_map(str(tmp_path / "m.json"))
    assert m[3] == "2" and m[1] == "10"
