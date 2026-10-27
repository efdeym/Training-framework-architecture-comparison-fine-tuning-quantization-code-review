"""exclude_train: listedeki dosya train'den çıkar, diğer sınıf/hedefler bozulmaz."""
from PIL import Image
from torchvision.datasets import ImageFolder
from flowers.data.datasets import drop_files


def test_drop_files(tmp_path):
    for c in ["1", "10", "2"]:
        (tmp_path / c).mkdir()
        for i in range(3):
            Image.new("RGB", (8, 8)).save(tmp_path / c / f"image_{c}_{i}.jpg")
    ds = ImageFolder(tmp_path)
    n = len(ds)
    drop_files(ds, ["image_10_1.jpg"])
    assert len(ds) == n - 1 and len(ds.targets) == n - 1 and len(ds.imgs) == n - 1
    assert all("image_10_1.jpg" not in p for p, _ in ds.samples)
    assert ds.class_to_idx == {"1": 0, "10": 1, "2": 2}
    drop_files(ds, [])
    assert len(ds) == n - 1