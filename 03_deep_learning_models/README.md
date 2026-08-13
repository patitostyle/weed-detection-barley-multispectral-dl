# 03 — Deep Learning Models

Trains and evaluates the three models compared for weed segmentation: U-Net, DeepLabv3 (ResNet-50 backbone), and Random Forest.

## Files

| Script | Model | Notes |
|---|---|---|
| `train_unet_base.py` | U-Net (custom, from scratch) | Baseline model. Best performer for the weed class. |
| `train_deeplabv3.py` | DeepLabv3 + ResNet-50 | First conv layer adapted to accept 8 input channels. |
| `train_random_forest.py` | Random Forest | Classical ML benchmark, trained on flattened pixel values. |

## Requirements

```
pip install torch torchvision rasterio numpy scikit-learn joblib
```

## Inputs

Patches generated in `02_patch_generation/` — a training set (e.g. Barley 1, flight V1) and a test/evaluation set (e.g. Barley 1, flight V2 or a different field), with matching `images/` and `masks/` folders.

## Outputs

Each script creates its own results folder containing:
- `best_model.pth` (U-Net / DeepLabv3) or `random_forest_model.joblib` (Random Forest)
- `summary.txt` with per-epoch (or final) metrics: F1 and IoU per class (crop, weed, soil) and mIoU

## How to run

1. Edit the **CONFIGURATION** section at the top of each script:
   - `TRAIN_IMG_DIR` / `TRAIN_MASK_DIR`: training patches
   - `TEST_IMG_DIR` / `TEST_MASK_DIR`: evaluation patches
   - `OUT_DIR`: where results are saved
2. Run:
   ```
   python train_unet_base.py
   python train_deeplabv3.py
   python train_random_forest.py
   ```

## Notes

- Mask labels are remapped internally: 1→crop, 2→weed, 3→soil, 0→ignored (255) during loss/metric computation.
- Each band is normalized per patch using the 2nd–98th percentile range.
- The weed class F1-score is used as the model selection criterion (`best_model.pth` is saved whenever it improves).
- These scripts were run once per input configuration (RGB / RGB+NIR+RedEdge / RGB+indices / RGB+NIR+RedEdge+indices) by changing `TRAIN_IMG_DIR`/`TEST_IMG_DIR` to the corresponding patch dataset, and once more for the spatial validation stage (training on Barley 1 V1+V2, evaluating on Barley 2).
- U-Net was ultimately selected as the baseline model for the modular hybrid approach (`04_hybrid_approach/`) and PSO optimisation (`05_metaheuristic_optimization/`).
