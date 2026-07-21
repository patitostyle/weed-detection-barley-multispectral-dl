# 02 — Patch Generation

Splits a multiband image and its reference mask into smaller, overlapping patches suitable for training semantic segmentation models.

## What it does

1. Opens the multiband input image (e.g. RGB + NIR + RedEdge + vegetation indices) and its corresponding single-band reference mask (crop / weed / soil labels).
2. Slides a 256×256 px window across both rasters with 50% overlap (stride = 128), keeping the image and mask perfectly aligned.
3. Discards patches that contain no valid class or are entirely background (value 0).
4. Saves each valid patch pair (image + mask) as separate GeoTIFF files with matching filenames.

## Requirements

```
pip install rasterio numpy
```

## Inputs

- A multiband raster (output of `01_gis_processing`, stacking RGB, NIR, RedEdge, and selected vegetation indices)
- A single-band raster mask with class labels (1 = crop, 2 = weed, 3 = soil, 0 = background), generated from manual labelling in ArcGIS Pro

## Outputs

```
output_path/
├── images/   # patch_00000.tif, patch_00001.tif, ...
└── masks/    # patch_00000.tif, patch_00001.tif, ... (same filenames, aligned with images)
```

## How to run

1. Edit the **CONFIGURATION** section at the top of `generate_patches.py`:
   - `image_path`: path to the multiband input raster
   - `mask_path`: path to the reference mask raster
   - `output_path`: where patches should be saved
   - `patch_size` / `stride`: patch dimensions and overlap (default: 256 px, 50% overlap)
   - `valid_classes`: class values that make a patch worth keeping
2. Run:
   ```
   python generate_patches.py
   ```

## Notes

- This script was run once per field/flight combination, producing separate patch datasets (e.g. Barley 1-V1, Barley 1-V2, Barley 2-V2, Oilseed rape-V2) that were later combined or kept separate depending on the training/evaluation stage (temporal validation, spatial validation, cross-crop transfer).
- `require_class = True` ensures the dataset isn't dominated by empty/background patches, keeping only patches that contain at least one of crop, weed, or soil.
