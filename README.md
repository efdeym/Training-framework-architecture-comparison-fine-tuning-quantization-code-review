# Flower Classifier (102 Classes)

Flower species classifier. Evaluation criteria: **top-1 accuracy (hidden test)**, **ms/image**, **MB**.
Instructor's priority: **a model that runs in the shortest time and occupies the least space**. Therefore, the MobileNet family was selected as the delivery candidate.

## Course and Team

- **Course:** Introduction to Generative AI (SENG 481)
- **Instructor:** Yüksel Arslan

| Name Surname | Student ID | Role |
|---|---|---|
| Elif Dilbeste Yıldırım | 220201021 | Technical Lead (model and training) |
| Karya Demirciler | 220201050 | Data and evaluation |
| Rabia Sude Karakoç | 230204045 | Inference, measurement, and compression |
| Hayrunnisa Öztürk | 220204048 | Environment, documentation, and quality |

Layered architecture and module purposes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Team plan and Git rules: [`docs/TEAM.md`](docs/TEAM.md).

## Status

| Stage | Status |
|---|---|
| Setup, tests | Completed |
| Training on real data & fair benchmark (Small, Large, EffNet-B0, ConvNeXt-nano) | Completed |
| LP-FT, LP epoch count, test time input size, quantization decision | Completed |
| Export, embedding normalization inside the model | Completed |
| EMA (0.999) | Completed: 1-2 images difference vs. baseline, curve is smoother, to be used in final_fit (A27) |
| final_fit training and final `deliver/flower_model.pt` | Completed: sanity check valid 817/818 (99.88%), test 819/819 (100%) |
| End-to-end latency measurement (including preprocessing) | Completed: 28.51 ms/image (CPU, 1 thread, batch=1) |
| Clean environment test (new venv: torch, numpy, Pillow) | Completed: 5/5 correct predictions, no `timm`/`flowers` dependencies |
| Shadow test | Completed: 306 images across 102 classes verified, domain shift documented (docs/shadow_test_report.md) |
| Organizer questions | No response received, assumptions listed below |

## Results

Only measured values are reported. Accuracies obtained via `evaluate.py` (valid 818, test 819 images). Latency: CPU, batch=1, model forward pass only, measured on development machine. In validation, 1 image = 0.12%.

| Model | Training | Checkpoint | Valid | Test | CPU ms (1 thr / multi-thr) | MB |
|---|---|---|---|---|---|---|
| `mobilenetv3_small_100` | 10 epochs, backbone frozen | - | 93.03% | 92.31% | 3.2 / - | 6.50 |
| `mobilenetv3_small_100` | 30 epochs, LP-FT | `last.pt` | 98.04% | 97.44% | 3.05 / 7.76 | 6.50 |
| **`mobilenetv3_large_100`** | 30 epochs, LP-FT | `best.pt` | **99.39%** | **99.27%** | **8.39** / 15.64 | 17.33 |
| `efficientnet_b0` | 30 epochs, LP-FT | - | 99.14% | 99.02% | 16.40 / 19.53 | 16.53 |
| `convnext_nano` | 30 epochs, LP-FT | - | 99.39% | 99.39% | 52.47 / - | 60.10 |

**How to interpret**
- The 2-image accuracy difference between Large and EffNet is within the noise margin; the speed difference (8.4 vs 16.4 ms) is decisive.
- ConvNeXt was eliminated due to size (60 MB) and speed.
- Recommended limits (50 ms, 35 MB) are not finalized; Large operates well below them.
- At batch=1, multi-threading is slower than single-threading.
- Validation/test sets belong to this dataset; they indicate relative model comparison rather than absolute success on the hidden test set.
- The source for the Small LP-FT row is `last.pt`. `runs/mnv3_small_ft/best.pt` was overwritten by a second (interrupted) run, making it invalid and unusable.

### EMA Comparison (Large, same recipe)

| Run | Checkpoint | Valid | Test |
|---|---|---|---|
| Baseline (`mobilenetv3_full`) | `best.pt` | 99.39% | 99.27% |
| Baseline (`mobilenetv3_full`) | `last.pt` | 99.27% | 99.39% |
| EMA 0.999 (`mnv3_large_ema`) | `best.pt` | 99.27% | 99.51% |
| EMA 0.999 (`mnv3_large_ema`) | `last.pt` | 99.27% | 99.51% |

The difference is 1-2 images, well within the noise margin. Since EMA yields identical results for `best.pt` and `last.pt`, using `last.pt` in final_fit is safe. EMA has zero runtime cost and smooths out the learning curve.

### Optimization Experiments (including unsuccessful ones)

| Experiment | Result | Decision |
|---|---|---|
| Backbone frozen vs LP-FT (Small) | 93.0% → 98.0% | LP-FT is used |
| LP epochs 3 vs 0 (Large) | 99.27% vs 99.14%, 1-image difference | Keep LP=3 |
| Upscaling input size at test time (256/288/320) | Accuracy dropped | Stayed at 224 |
| Multi-threading (batch=1) | 15.6 ms vs 8.4 ms (1 thread) | No extra tuning applied |
| fp16 / int8 quantization | Speed and size already well under thresholds | Skipped |
| TTA (flips, etc.) | Not measured, doubles inference time | Not added |
| Large + EffNet ensemble | 33.9 MB, 1.1 MB below the 35 MB limit | Dropped (risky) |
| EMA (0.999) | Equivalent to baseline on valid/test, smoother curve | Used in final_fit |

### Preprocessing Robustness (valid, 818 images, Large)

| Variant | Accuracy |
|---|---|
| Default (resize + center crop) | 99.39% |
| Squeezing to 224x224 | 98.90% |
| Blur (sigma=2) | 98.29% |
| Darkening (x0.6) | 98.90% |
| Brightening (x1.4) | 98.41% |
| JPEG quality 20 | 98.66% |

Script: `scripts/robustness_probe.py`.

### End-to-End Latency (including preprocessing)

Measured for `deliver/flower_model.pt` (Large): **28.51 ms/image** (includes image loading, resizing, and normalization; CPU, 1 thread, batch=1, 200 test images). Since the model forward pass alone takes 8.39 ms, most of the time is spent on preprocessing. Both values are below the 50 ms recommended limit.

Command: `python scripts\e2e_time.py deliver\flower_model.pt`

**MB Definition:** file size / 10⁶ bytes. By this definition, `flower_model.pt` = **17.33 MB** (16.52 MiB using 1024²). All tables above use the same definition.

## Setup

```bash
python -m venv .venv
# Windows (cmd):         .venv\Scripts\activate.bat
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Mac/Linux:             source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # or: PYTHONPATH=src
pytest -q tests           # 3 tests should pass
