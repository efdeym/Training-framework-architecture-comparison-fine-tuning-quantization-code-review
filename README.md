# Flower Classifier (102 classes)

Flower species classifier. Evaluation criteria: top-1 accuracy (hidden test), ms/image, MB.
Priority: Lightest and fastest model. MobileNet family was chosen as the primary candidate.


## Course and Team

- **Course:** Introduction to Generative AI (SENG 481)
- **Instructor:** Yüksel Arslan

| Name Surname | Student ID | Role |
|---|---|---|
| Elif Dilbeste Yıldırım | 220201021 | Technical Lead (model and training) |
| Karya Demirciler | 220201050 | Data and evaluation |
| Rabia Sude Karakoç | 230204045 | Inference, measurement, and compression |
| Hayrunnisa Öztürk | 220204048 | Environment, documentation, and quality |

Layered architecture and class purposes: `docs/ARCHITECTURE.md`. Team plan and Git rules: `docs/TEAM.md`.

## Status

| Stage | Status |
|---|---|
| Setup, tests | Done |
| Training on real data and fair comparison (Small, Large, EffNet-B0, ConvNeXt-nano) | Done |
| LP-FT, number of LP epochs, test-time input size, quantization decision | Done |
| Export, embedding normalization into the model | Done |
| EMA (0.999) | Done: 1-2 images difference vs. baseline, smoother curve, used in final_fit (A27) |
| final_fit training and final `deliver/flower_model.pt` | Done: sanity check valid 817/818 (99.88%), test 819/819 (100%) |
| End-to-end timing (including preprocessing) | Done: 28.51 ms/image (CPU, 1 thread, batch=1) |
| Clean-environment test (new venv: torch, numpy, Pillow) | Done: 5/5 correct predictions, no timm/flowers dependency |
| `submission/model.pt` in the instructor's format (softmax + class order inside the model) | Done: `check_submission.py` passed (bilinear and bicubic), 17.33 MB |
| External data test (non-Oxford photos) | Completed: TensorFlow set 35–39%, Wikimedia 59–63%; see External Data Test |


## Results

Measured values only. Accuracies computed with `evaluate.py` (valid: 818, test: 819 images). Latency: CPU, batch=1, model forward pass only, on the development computer. On valid, 1 image = 0.12%.

| Model | Training | Checkpoint | Valid | Test | CPU ms (1 thr / multi thr) | MB |
|---|---|---|---|---|---|---|
| mobilenetv3_small_100 | 10 epochs, frozen backbone | - | 93.03% | 92.31 | 3.2 / - | 6.50 |
| mobilenetv3_small_100 | 30 epochs, LP-FT | last.pt | 98.04% | 97.44 | 3.05 / 7.76 | 6.50 |
| mobilenetv3_large_100 | 30 epochs, LP-FT | best.pt | 99.39% | 99.27 | 8.39 / 15.64 | 17.33 |
| efficientnet_b0 | 30 epochs, LP-FT | - | 99.14% | 99.02 | 16.40 / 19.53 | 16.53 |
| convnext_nano | 30 epochs, LP-FT | - | 99.39% | 99.39 | 52.47 / - | 60.10 |

### How to read

- The 2-image accuracy gap between Large and EffNet is at the noise level; the speed difference (8.4 vs 16.4 ms) is the deciding factor.
- ConvNeXt was eliminated because of size (60 MB) and speed.
- The suggested limits (50 ms, 35 MB) are not finalized; Large stays well below them.
- At batch=1, multi-threading is slower than a single thread.
- Valid/test come from this dataset; for the hidden test they show not absolute performance but the comparison between models.
- The source of the Small LP-FT row is `last.pt`. `runs/mnv3_small_ft/best.pt` was overwritten by a second (interrupted) run, so it is invalid and must not be used.

### EMA comparison (Large, same recipe)

| Run | Checkpoint | Valid | Test |
|---|---|---|---|
| Baseline (mobilenetv3_full) | best.pt | 99.39% | 99.27% |
| Baseline (mobilenetv3_full) | last.pt | 99.27% | 99.39% |
| EMA 0.999 (mnv3_large_ema) | best.pt | 99.27% | 99.51% |
| EMA 0.999 (mnv3_large_ema) | last.pt | 99.27% | 99.51% |

The difference is 1-2 images, i.e. at the noise level. Since EMA's `best.pt` and `last.pt` results are identical, using `last.pt` in final_fit is safe. EMA is free and smooths the curve.

### Optimization experiments (including those that didn't help)

| Experiment | Result | Decision |
|---|---|---|
| Frozen backbone vs LP-FT (Small) | 93.0% → 98.0% | LP-FT is used |
| LP epochs 3 vs 0 (Large) | 99.27% vs 99.14%, 1 image difference | LP=3 stays |
| Test-time input enlargement (256/288/320) | Accuracy dropped | Stayed at 224 |
| Multi-thread (batch=1) | 15.6 ms vs 8.4 ms (1 thread) | No further tuning |
| fp16 / int8 quantization | Speed and size already below limits | Skipped |
| TTA (flip etc.) | Not measured, doubles the time | Not added |
| Large + EffNet ensemble | 33.9 MB, 1.1 MB below the 35 MB limit | Dropped (risk) |
| EMA (0.999) | Equivalent to baseline on valid/test, smooth curve | Used in final_fit |

### Preprocessing robustness (valid, 818 images, Large)

| Variant | Accuracy |
|---|---|
| Default (resize + center crop) | 99.39% |
| Squash to 224x224 | 98.90% |
| Blur (sigma=2) | 98.29% |
| Dark (x0.6) | 98.90% |
| Bright (x1.4) | 98.41% |
| JPEG quality 20 | 98.66% |

Script: `scripts/robustness_probe.py`.

### End-to-end timing (including preprocessing)

Measurement for `deliver/flower_model.pt` (Large): 28.51 ms/image (includes image reading, resizing and normalization; CPU, 1 thread, batch=1, 200 test images). Since the model forward pass alone is 8.39 ms, most of the time is spent on preprocessing. Both values are below the suggested 50 ms limit.

Command: `python scripts\e2e_time.py deliver\flower_model.pt`

MB definition: file size / 10⁶ bytes. With this definition `flower_model.pt` = 17.33 MB (16.52 MiB using 1024²). All tables above use the same definition.

## External Data Test (Non-Oxford Photos)

Udacity's `flower_data` set matched Oxford 1-to-1 (8,189/8,189 files), so it was not treated as independent data. Two independent sources were evaluated using instructor preprocessing (short side 256, center crop 224, range 0-1) across 102 classes (`scripts/hoca_eval.py`):

- **TensorFlow flower_photos:** 3 overlapping classes with Oxford (dandelion → 50, roses → 74, sunflowers → 54), 2,238 photos.
- **Wikimedia Commons (`scripts/fetch_shadow.py`):** 28 classes, 224 photos, uncurated (contains label noise).

Models that did not see valid/test during training were compared on these datasets:

| Model | TensorFlow (2,238) | Wikimedia (224) | Model Time (ms) | MB |
|---|---|---|---|---|
| mobilenetv3_large_100 baseline | 35.75% | 59.38% | 9.9 | 17.33 |
| mobilenetv3_large_100 + EMA | 38.70% | 59.82% | 9.0 | 17.33 |
| efficientnet_b0 | 35.48% | 58.93% | 16.5 | 16.53 |
| convnext_nano | 38.65% | 63.39% | 54.6 | 60.10 |
| **Submission model (final_fit)** | **36.95%** | **59.82%** | **-** | **17.33** |

### Findings

- All four architectures fall within the same performance band; changing the architecture does not bridge the gap. The drop is caused by the shift in photographic style between training and test sets.
- Per-class breakdown (EMA model): sunflower 376/699, rose 328/641, common dandelion 162/898.
- Analysis of the top 16 sunflower errors: wide shots (e.g., fields), small flowers inside vases/scenes, unusual color varieties, and unrelated photos (label noise). Errors stem both from the model's sensitivity to uncentered compositions and noise in the test set.
- Dandelion errors mostly go to prickly/round-headed classes (globe thistle, spear thistle, alpine sea holly); the TensorFlow "dandelion" folder contains seed head photos, reflecting a partial semantic mismatch.

## Setup

```bash
python -m venv .venv
# Windows (cmd):         .venv\Scripts\activate.bat
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Mac/Linux:             source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # or: PYTHONPATH=src
pytest -q tests           # 3 tests should pass
```

Python 3.10 or higher is required.

## Data

```
data/flowers/
  train/<class_no>/*.jpg    (6,552 images)
  valid/<class_no>/*.jpg    (818)
  test/<class_no>/*.jpg     (819)
```

102 classes, the same classes in all three splits. Train has 27-206 images per class (imbalanced), valid has 1-28 images per class, so valid accuracy is somewhat volatile.

Folder names are numeric text (1, 10, 100...). ImageFolder indexes them alphabetically; the mapping is automatically saved into the checkpoint and the submission file, and `predict` returns the original folder name.

Extracting the data from the archive (Windows):

```bash
mkdir data\flowers
tar -xzf flower_data.tar.gz -C data\flowers
```

`data/` and `*.tar.gz` are not committed to Git (`.gitignore`).

## Quick Start

```bash
# 1) Baseline training (MobileNetV3-Small, frozen backbone, 10 epochs)
python scripts/train.py --config configs/mnv3_small.yaml data.root=data/flowers

# 2) Another architecture / override
python scripts/train.py --config configs/mobilenetv3.yaml data.root=data/flowers train.epochs=10 train.linear_probe_epochs=10 exp_name=mnv3_large_lp10

# 3) Produce the submission file (single file, timm not required)
python scripts/export.py --ckpt runs/mnv3_small/best.pt --out mnv3_small.pt --dtype fp32

# 4) Measure speed + size and add to the table (write the real value from training instead of --top1)
python scripts/benchmark.py --model mnv3_small.pt --device cpu --threads 1 --tag mnv3_small --top1 0.9303
python scripts/pareto.py       # results/pareto.png

# 5) Shadow test (images collected from a different source, same folder names)
python scripts/evaluate.py --ckpt runs/mnv3_small/best.pt --data data/shadow_test
```

## Experiment Order

1. Baseline with `mnv3_small` and `mobilenetv3` (Large), same settings, Pareto chart
2. Small and fast candidates: `mobilenetv3_small_050`, `mobilenetv2_100`, `mobilenetv4_conv_small`
3. Resolution (160/192/224) and recipe (LP-FT, EMA) experiments
4. fp16 / int8 compression (`flowers.compression.quantize`), measuring accuracy and MB at each step
5. Optional: distillation with a large teacher (e.g. ResNet50) (`configs/distill_student.yaml`)
6. Last: `configs/final_fit.yaml` (train+valid+test combined), export with `last.pt`

**final_fit decision (A27):** The selected recipe (Large, LP-FT, EMA 0.999, 224, 30 epochs, seed 42) is retrained once on the train+valid+test union (+25% data). Accuracy measured on valid/test after this training is meaningless (the data was seen during training); only a sanity check is done (~100% expected). Sanity check result (`scripts/check_deliver.py`): valid 99.88%, test 100%; these numbers only show that the model was loaded correctly and are not reported as generalization performance. Backups: `runs/mnv3_large_ema` and `runs/mobilenetv3_full`. The gain is small (a few images on the hidden test); the risk is limited by the sanity check and the backup.

**About ResNet50:** At roughly 95-100 MB it is not suitable as the submission model. It can be used as a comparison reference or as a distillation teacher.

**Selection rule (proposal):** Among the models whose shadow-test accuracy does not fall below the accuracy floor set by the team, the one with the smallest MB and lowest ms is selected.

## Submission: model.pt

The instructor's rule: a single file named `model.pt`, TorchScript (architecture + weights + normalization inside), runs on CPU. No other file (e.g. `predict.py`) is used.

| Rule | Ours |
|---|---|
| Input | `[B,3,224,224]`, RGB, range 0-1; the evaluator does not apply mean/std (short side 256 + center crop 224) |
| Normalization | Inside the model (ImageNet mean/std) |
| Output | `[B,102]` softmax probabilities, each row sums to 1 |
| Output order | Position 0 = class 1, ..., position 101 = class 102. ImageFolder's alphabetical order is converted to numerical order inside the model |
| File size | 17.33 MB (10⁶ bytes) |

Producing it (source: `deliver/flower_model.pt`, the final_fit model):

```bash
python scripts/make_submission.py      # produces submission/model.pt
python scripts/check_submission.py     # mimics the instructor's preprocessing, checks shape/range/sum/order
```

Validation result (since the final_fit model has seen valid/test, these numbers only show that the order and softmax are correct): with bilinear and bicubic resizing, valid 817/818, test 819/819.

Usage:

```python
import torch
model = torch.jit.load("model.pt", map_location="cpu").eval()
# x: [B,3,224,224], RGB, in the 0-1 range, NOT normalized
probs = model(x)                 # [B,102], softmax
class_no = probs.argmax(1) + 1   # 1..102 (folder number)
```

Note: `torch.jit.load` gives a "deprecated" warning in newer PyTorch versions; it still works. Tested in a clean venv (torch 2.15, numpy 2.5, Pillow 13, Python 3.12).

## Evaluation Conditions

The instructor's submission rules are finalized (see the "Submission" section above). Items in the guide: Intel i7, 16 GB, no GPU; score = 0.60 x accuracy + 0.25 x speed + 0.15 x size; Tmax=50 ms, Smax=35 MB (suggested, not finalized).

Since the instructor's measurement method (number of threads, whether preprocessing is included in the time) is unknown, the following values are reported:

- Model forward pass only: 8.39 ms (1 thread), 15.64 ms (14 threads)
- End-to-end including preprocessing: 28.51 ms (1 thread)

## Known Risks and Notes

- **Domain shift (measured):** Accuracy of 99%+ on Oxford valid/test drops significantly on non-Oxford photos (see External Data Test). If the hidden test consists of centered close-ups similar to Oxford, performance will remain high; otherwise, lower results are expected.
- **No hue augmentation:** Color determines the flower species.
- **fp16:** Fast and half the size on GPU; on CPU `predict.py` automatically converts to fp32. Verify the fp16 TorchScript path separately on the target hardware.
- **int8 (FX PTQ):** Try it if targeting CPU; it may cause problems on a per-model basis, always re-measure accuracy.
- Pretrained weights are downloaded from the internet (Hugging Face Hub) via timm; a connection is needed on first run. The "symlinks" warning on Windows is harmless.
- **torch.load FutureWarning:** The `torch.load` call in `packager.py` gives a `weights_only` warning; it is harmless since we only load our own checkpoints.
- **final_fit:** Accuracy measured on this data after the combined training is meaningless; the decision is made with the shadow test.

## Tests

```bash
pytest -q tests
```

Tests: label mapping, consistency of `predict.py` preprocessing with the training eval transform, LR scheduler.