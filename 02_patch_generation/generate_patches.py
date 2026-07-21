"""
Generate image/mask patches for semantic segmentation training.

Splits a multiband raster image and its corresponding reference mask into
smaller, overlapping patches (e.g. 256x256 px), preserving spatial
correspondence between each image patch and its mask. Patches that contain
no valid class (or are entirely background) are discarded.

Requires: rasterio, numpy
"""

import os
import rasterio
from rasterio.windows import Window
import numpy as np

# =========================================================
# CONFIGURATION — EDIT THIS SECTION FOR EACH FIELD/FLIGHT
# =========================================================

# Path to the multiband input raster (e.g. RGB + NIR + RedEdge + indices)
image_path = r"PATH_TO_MULTIBAND_IMAGE.tif"

# Path to the corresponding single-band reference mask
mask_path = r"PATH_TO_REFERENCE_MASK.tif"

# Output folder where patches will be saved
output_path = r"PATH_TO_OUTPUT_FOLDER"

# Patch size (pixels) and stride (overlap = patch_size - stride)
patch_size = 256
stride = 128

# Class values considered valid (e.g. 1 = crop, 2 = weed, 3 = soil)
valid_classes = [1, 2, 3]

# If True, a patch is only kept when it contains at least one valid class
require_class = True

# =========================================================
# DO NOT EDIT BELOW UNLESS CHANGING THE OUTPUT STRUCTURE
# =========================================================

output_images = os.path.join(output_path, "images")
output_masks = os.path.join(output_path, "masks")

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# =========================================================
# OPEN RASTERS
# =========================================================
with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:

    print("=== Initial check ===")
    print("Image:", src_img.width, src_img.height, "bands:", src_img.count)
    print("Mask:", src_mask.width, src_mask.height, "bands:", src_mask.count)

    # Use the smallest common size between image and mask
    width = min(src_img.width, src_mask.width)
    height = min(src_img.height, src_mask.height)

    print("\n=== Common size used ===")
    print("Columns:", width)
    print("Rows:", height)

    patch_count = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):

            window = Window(x, y, patch_size, patch_size)

            img_patch = src_img.read(window=window)      # (bands, h, w)
            mask_patch = src_mask.read(1, window=window)  # (h, w)

            # Check whether the patch contains any valid class
            patch_values = np.unique(mask_patch)
            has_valid_class = np.any(np.isin(patch_values, valid_classes))

            if require_class and not has_valid_class:
                continue

            if np.all(mask_patch == 0):
                continue

            filename = f"patch_{patch_count:05d}.tif"
            img_out_path = os.path.join(output_images, filename)
            mask_out_path = os.path.join(output_masks, filename)

            img_profile = src_img.profile.copy()
            img_profile.update({
                "height": patch_size,
                "width": patch_size,
                "transform": rasterio.windows.transform(window, src_img.transform)
            })

            mask_profile = src_mask.profile.copy()
            mask_profile.update({
                "height": patch_size,
                "width": patch_size,
                "count": 1,
                "transform": rasterio.windows.transform(window, src_mask.transform)
            })

            with rasterio.open(img_out_path, "w", **img_profile) as dst_img:
                dst_img.write(img_patch)

            with rasterio.open(mask_out_path, "w", **mask_profile) as dst_mask:
                dst_mask.write(mask_patch, 1)

            patch_count += 1

print(f"\nTotal patches saved: {patch_count}")
print("Images folder:", output_images)
print("Masks folder:", output_masks)
