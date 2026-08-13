# 04 — Modular Hybrid Approach

Refines the baseline U-Net predictions by applying spatial/agronomic post-processing rules based on crop-row position, detection confidence, and object size.

## Files

| Script | Rule set |
|---|---|
| `hybrid_v1.py` | Strict: confidence threshold 0.70 inside crop row, removes small objects (min size 4) |
| `hybrid_v2.py` | Softer: confidence threshold 0.60 inside crop row, lighter cleanup (min size 2) |
| `hybrid_v3.py` | Object-level: evaluates each detected object's mean confidence, size, and proportion inside the crop row before keeping or discarding it |

All three load the same pretrained U-Net baseline (`03_deep_learning_models/train_unet_base.py` output) and only change the post-processing logic — no retraining involved.

## Requirements

```
pip install torch rasterio numpy scipy
```

## Inputs

- Test patches (`images/`, `masks/`) from `02_patch_generation/`
- A matching **row mask** folder (`rowmask/`) — binary raster patches (1 = inside crop row, 0 = outside), generated from crop-row guide lines manually digitised in ArcGIS Pro, buffered and rasterised
- The trained U-Net weights (`best_model.pth`) from `03_deep_learning_models/`

## Outputs

Each script saves a `hybrid_summary.txt` with:
- The configuration/thresholds used
- Baseline U-Net metrics (F1, IoU per class, mIoU)
- Hybrid-adjusted metrics (F1, IoU per class, mIoU)

## How to run

1. Edit the **CONFIGURATION** section at the top of each script (`IMG_DIR`, `MASK_DIR`, `ROWMASK_DIR`, `MODEL_PATH`, `OUT_DIR`)
2. Run:
   ```
   python hybrid_v1.py
   python hybrid_v2.py
   python hybrid_v3.py
   ```

## Notes

- None of the three variants improved on the baseline U-Net in this study — differences in weed F1/IoU were small (< 0.005) across all versions. See the main repo README for the full comparison table.
- The rules operate on the weed-class probability map and the row mask; predictions for crop and soil are otherwise left as the model's base output, except pixels previously classified as weed that no longer meet the hybrid criteria (which are reassigned to crop).
