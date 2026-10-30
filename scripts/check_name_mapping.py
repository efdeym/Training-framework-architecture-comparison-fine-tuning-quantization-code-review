"""Validate cat_to_name.json class IDs against embedded TorchScript class map.

Run from repository root:
  python scripts/check_name_mapping.py --model flower_model_fp32.pt --names cat_to_name.json
This checks coverage, not botanical correctness; visually spot-check flower names.
"""
import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flower_model_fp32.pt")
    parser.add_argument("--names", default="cat_to_name.json")
    args = parser.parse_args()

    with open(args.names, encoding="utf-8-sig") as stream:
        names = json.load(stream)
    if not isinstance(names, dict):
        raise ValueError("Name file must contain a JSON object of class_id -> flower name")

    extra = {"meta.json": ""}
    torch.jit.load(args.model, map_location="cpu", _extra_files=extra)
    if not extra["meta.json"]:
        raise ValueError("The model has no embedded meta.json; use the TorchScript export.")
    metadata = json.loads(extra["meta.json"])
    expected = {str(value) for value in metadata["idx_to_class"].values()}
    actual = {str(key) for key in names}
    missing = sorted(expected - actual, key=lambda x: int(x) if x.isdigit() else x)
    extras = sorted(actual - expected, key=lambda x: int(x) if x.isdigit() else x)
    blanks = sorted((str(k) for k, v in names.items() if not isinstance(v, str) or not v.strip()))
    print(f"Model classes: {len(expected)} | Name entries: {len(actual)}")
    print(f"Missing IDs: {missing}")
    print(f"Extra IDs: {extras}")
    print(f"Empty/invalid names: {blanks}")
    print(f"Class 1: {names.get('1', '(missing)')}")
    print(f"Class 102: {names.get('102', '(missing)')}")
    if missing or extras or blanks:
        raise SystemExit("Mapping check FAILED. Fix IDs or names before using name output.")
    print("Mapping IDs OK. NOTE: verify botanical names by examining sample photos.")


if __name__ == "__main__":
    main()
