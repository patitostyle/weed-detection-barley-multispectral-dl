"""
Clip UAV multispectral bands to the study field and calculate vegetation indices.

Workflow:
1. Clip the 5 raw spectral bands (Blue, Green, Red, NIR, RedEdge) to the field
   boundary polygon, ensuring consistent extent, cell size, and alignment
   across all bands (same path/row).
2. Build an RGB composite from the clipped bands.
3. Calculate vegetation indices: NDVI, NDRE, VARI, SAVI, GNDVI, CCCI.

Requires ArcGIS Pro with the Spatial Analyst extension (arcpy.sa).
"""

import os
import arcpy
from arcpy.sa import ExtractByMask, Float

# =========================================================
# CONFIGURATION — EDIT THIS SECTION FOR EACH FIELD/FLIGHT
# =========================================================

# Folder containing the 5 ORIGINAL raw spectral bands
input_folder = r"PATH_TO_RAW_BANDS_FOLDER"

# Field boundary polygon (shapefile) used to clip the imagery
field_boundary = r"PATH_TO_FIELD_BOUNDARY.shp"

# Root output folder
output_folder = r"PATH_TO_OUTPUT_FOLDER"

# Short field identifier, used to name output files (e.g. "barley2_v1")
field_id = "barley2_v1"

# Exact input band filenames
blue_file = "blue.tif"
green_file = "green.tif"
red_file = "red.tif"
nir_file = "nir.tif"
rededge_file = "rededge.tif"

# =========================================================
# DO NOT EDIT BELOW UNLESS CHANGING THE OUTPUT STRUCTURE
# =========================================================

# Create organized output subfolders
clip_folder = os.path.join(output_folder, "01_clipped_bands")
rgb_folder = os.path.join(output_folder, "02_rgb")
indices_folder = os.path.join(output_folder, "03_indices")

for folder in [output_folder, clip_folder, rgb_folder, indices_folder]:
    os.makedirs(folder, exist_ok=True)

# Full input paths
blue_path = os.path.join(input_folder, blue_file)
green_path = os.path.join(input_folder, green_file)
red_path = os.path.join(input_folder, red_file)
nir_path = os.path.join(input_folder, nir_file)
rededge_path = os.path.join(input_folder, rededge_file)

# Clipped band output paths
blue_clip = os.path.join(clip_folder, f"blue_{field_id}_clip.tif")
green_clip = os.path.join(clip_folder, f"green_{field_id}_clip.tif")
red_clip = os.path.join(clip_folder, f"red_{field_id}_clip.tif")
nir_clip = os.path.join(clip_folder, f"nir_{field_id}_clip.tif")
rededge_clip = os.path.join(clip_folder, f"rededge_{field_id}_clip.tif")

# RGB composite output
rgb_clip = os.path.join(rgb_folder, f"rgb_{field_id}_clip.tif")

# Vegetation index output paths
ndvi_out = os.path.join(indices_folder, f"NDVI_{field_id}_clip.tif")
ndre_out = os.path.join(indices_folder, f"NDRE_{field_id}_clip.tif")
vari_out = os.path.join(indices_folder, f"VARI_{field_id}_clip.tif")
savi_out = os.path.join(indices_folder, f"SAVI_{field_id}_clip.tif")
gndvi_out = os.path.join(indices_folder, f"GNDVI_{field_id}_clip.tif")
ccci_out = os.path.join(indices_folder, f"CCCI_{field_id}_clip.tif")

# ArcPy environment settings
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")


# =========================================================
# HELPER FUNCTION
# =========================================================
def check_exists(path):
    if not arcpy.Exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


# Validate all inputs before processing
for path in [blue_path, green_path, red_path, nir_path, rededge_path, field_boundary]:
    check_exists(path)

# =========================================================
# 1. CLIP BANDS TO THE FIELD BOUNDARY
# =========================================================
# Use the Red band as the reference for snap raster and cell size,
# ensuring all clipped bands share the same extent and alignment (path/row)
arcpy.env.snapRaster = red_path
arcpy.env.cellSize = red_path
arcpy.env.extent = field_boundary

print("Clipping bands...")

ExtractByMask(blue_path, field_boundary).save(blue_clip)
ExtractByMask(green_path, field_boundary).save(green_clip)
ExtractByMask(red_path, field_boundary).save(red_clip)
ExtractByMask(nir_path, field_boundary).save(nir_clip)
ExtractByMask(rededge_path, field_boundary).save(rededge_clip)

print("Bands clipped.")

# =========================================================
# 2. BUILD RGB COMPOSITE
# =========================================================
print("Creating RGB composite...")

arcpy.CompositeBands_management(
    in_rasters=f"{red_clip};{green_clip};{blue_clip}",
    out_raster=rgb_clip
)

print("RGB composite created.")

# =========================================================
# 3. CALCULATE VEGETATION INDICES
# =========================================================
print("Calculating vegetation indices...")

red_r = arcpy.Raster(red_clip)
green_r = arcpy.Raster(green_clip)
blue_r = arcpy.Raster(blue_clip)
nir_r = arcpy.Raster(nir_clip)
rededge_r = arcpy.Raster(rededge_clip)

# NDVI = (NIR - Red) / (NIR + Red)
ndvi = Float(nir_r - red_r) / Float(nir_r + red_r)
ndvi.save(ndvi_out)

# NDRE = (NIR - RedEdge) / (NIR + RedEdge)
ndre = Float(nir_r - rededge_r) / Float(nir_r + rededge_r)
ndre.save(ndre_out)

# VARI = (Green - Red) / (Green + Red - Blue)
vari = Float(green_r - red_r) / Float(green_r + red_r - blue_r)
vari.save(vari_out)

# SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5)
savi = 1.5 * Float(nir_r - red_r) / Float(nir_r + red_r + 0.5)
savi.save(savi_out)

# GNDVI = (NIR - Green) / (NIR + Green)
gndvi = Float(nir_r - green_r) / Float(nir_r + green_r)
gndvi.save(gndvi_out)

# CCCI = NDRE / NDVI
ccci = Float(
    (Float(nir_r - rededge_r) / Float(nir_r + rededge_r)) /
    (Float(nir_r - red_r) / Float(nir_r + red_r))
)
ccci.save(ccci_out)

print("Vegetation indices calculated.")

# =========================================================
# 4. SUMMARY
# =========================================================
print("\nProcess finished.")
print("Outputs saved to:")
print("Clipped bands:", clip_folder)
print("RGB composite:", rgb_clip)
print("Indices:", indices_folder)

arcpy.CheckInExtension("Spatial")
