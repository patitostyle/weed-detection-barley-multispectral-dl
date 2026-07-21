# 01 — GIS Processing

Clips UAV multispectral orthomosaic bands to the study field boundary and calculates vegetation indices, using ArcGIS Pro (arcpy + Spatial Analyst).

## What it does

1. **Clips** the 5 raw spectral bands (Blue, Green, Red, NIR, RedEdge) to the field boundary polygon, using the Red band as the reference for snap raster and cell size. This guarantees all output bands share the same extent, resolution, and pixel alignment (path/row).
2. **Builds an RGB composite** from the clipped Red, Green, and Blue bands.
3. **Calculates 6 vegetation indices** from the clipped bands: NDVI, NDRE, VARI, SAVI, GNDVI, CCCI.

## Requirements

- ArcGIS Pro with a licensed **Spatial Analyst** extension
- Python environment bundled with ArcGIS Pro (arcpy is not available via pip)

## Inputs

- 5 single-band raster files (Blue, Green, Red, NIR, RedEdge) from the UAV multispectral sensor, already orthorectified (e.g. via Pix4D or Agisoft Metashape)
- A field boundary polygon (shapefile) for clipping

## Outputs

Organized automatically into subfolders under the configured output path:

```
output_folder/
├── 01_clipped_bands/   # 5 clipped single-band rasters
├── 02_rgb/             # RGB composite
└── 03_indices/         # NDVI, NDRE, VARI, SAVI, GNDVI, CCCI
```

## How to run

1. Open the script in the ArcGIS Pro Python environment (or run it through ArcGIS Pro's Python window)
2. Edit the **CONFIGURATION** section at the top of `clip_and_indices.py`:
   - `input_folder`: path to the folder with the 5 raw band files
   - `field_boundary`: path to the field boundary shapefile
   - `output_folder`: where results should be saved
   - `field_id`: short identifier used in output filenames (e.g. `barley2_v1`)
   - Band filenames if they differ from the defaults
3. Run the script

## Notes

- Index formulas used:
  - NDVI = (NIR − Red) / (NIR + Red)
  - NDRE = (NIR − RedEdge) / (NIR + RedEdge)
  - VARI = (Green − Red) / (Green + Red − Blue)
  - SAVI = 1.5 × (NIR − Red) / (NIR + Red + 0.5)
  - GNDVI = (NIR − Green) / (NIR + Green)
  - CCCI = NDRE / NDVI
- This same script was run once per field/flight combination (e.g. Barley 1-V1, Barley 1-V2, Barley 2-V2, Oilseed rape-V2), changing the configuration variables at the top each time.
