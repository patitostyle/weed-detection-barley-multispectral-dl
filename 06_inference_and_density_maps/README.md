# 06 — Inference & Density Maps

Applies the final selected model (U-Net) to a complete field raster through sliding-window spatial inference, then generates weed probability and density maps for agronomic interpretation.

## What it does

**Step 1 — Full-field inference:**
1. Loads the trained U-Net and slides a 256×256 window (128 px stride / 50% overlap) across the entire multiband raster of the evaluation field.
2. Averages predicted probabilities in overlapping areas to produce a smooth, continuous output.
3. Saves two rasters: a **class map** (1 = crop, 2 = weed, 3 = soil, 0 = no data) and a **continuous weed probability map**.

**Step 2 — Density mapping:**
1. Aggregates the pixel-level outputs into coarser cells (default 2×2 m).
2. Produces:
   - A **continuous probability density** raster (mean weed probability per cell, %)
   - A **binary density** raster (% of pixels classified as weed per cell)
   - A **reclassified density** raster into 4 relative infestation levels: very low (0–5%), low (5–15%), medium (15–30%), high (>30%)

## Requirements

```
pip install torch rasterio numpy
```

## Inputs

- The trained U-Net weights (`best_model.pth`) — ideally the model selected after spatial validation (`03_deep_learning_models/`) or PSO optimisation (`05_metaheuristic_optimization/`)
- A full multiband raster (same band configuration used in training — RGB + NIR + RedEdge + vegetation indices) covering the field to map, e.g. output of `01_gis_processing/`

## Outputs

```
results/
├── inference_<field>/
│   ├── pred_classes_<field>.tif       # 1 crop / 2 weed / 3 soil
│   ├── prob_weed_<field>.tif          # continuous weed probability
│   └── inference_info.json            # run configuration
└── density_<field>/
    ├── density_prob_weed_2m.tif       # mean weed probability per cell (%)
    ├── density_bin_weed_2m.tif        # % pixels classified as weed per cell
    ├── density_weed_2m_classes.tif    # reclassified into 4 infestation levels
    └── density_info.json              # run configuration + class legend
```

## How to run

1. Edit the **CONFIGURATION** section at the top of `inference_and_density_maps.py`:
   - `MODEL_PATH`: path to the trained model weights
   - `INPUT_RASTER`: full multiband raster of the field to map
   - `CELL_SIZE_METERS`: density aggregation cell size (default 2 m)
2. Run:
   ```
   python inference_and_density_maps.py
   ```
   This runs both steps (inference, then density mapping) sequentially.

## Notes

- Pixels with no valid signal in any band (outside the field boundary) are marked as no-data and excluded from both inference and density outputs.
- The class raster and probability raster share the same resolution as the input; the density rasters are aggregated to the configured cell size with an updated affine transform.
- These outputs correspond to the final probability and density maps presented in the main repo README (spatial validation on Barley 2).
