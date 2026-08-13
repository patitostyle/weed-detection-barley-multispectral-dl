import os
import json
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import torch
import torch.nn as nn

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.expanduser("~/TFM_WEEDS")

# Final selected model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "unet_train_v1v2_test_barley2_30ep",
    "best_model.pth"
)

# Full raster of the evaluation field (e.g. Barley 2)
INPUT_RASTER = os.path.join(
    BASE_DIR,
    "data",
    "patches",
    "full_rasters",
    "mb_rgb_nir_re_indices_barley2.tif"
)

# Outputs
OUT_DIR_INF = os.path.join(BASE_DIR, "results", "inference_barley2")
OUT_DIR_DENS = os.path.join(BASE_DIR, "results", "density_barley2")
os.makedirs(OUT_DIR_INF, exist_ok=True)
os.makedirs(OUT_DIR_DENS, exist_ok=True)

OUT_CLASS = os.path.join(OUT_DIR_INF, "pred_classes_barley2.tif")
OUT_PROB = os.path.join(OUT_DIR_INF, "prob_weed_barley2.tif")
OUT_INFO_INF = os.path.join(OUT_DIR_INF, "inference_info.json")

OUT_DENS_PROB = os.path.join(OUT_DIR_DENS, "density_prob_weed_2m.tif")
OUT_DENS_BIN = os.path.join(OUT_DIR_DENS, "density_bin_weed_2m.tif")
OUT_DENS_CLASS = os.path.join(OUT_DIR_DENS, "density_weed_2m_classes.tif")
OUT_INFO_DENS = os.path.join(OUT_DIR_DENS, "density_info.json")

# Model
IN_CHANNELS = 8
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference
PATCH_SIZE = 256
STRIDE = 128

# Density aggregation
CELL_SIZE_METERS = 2.0
NODATA_PROB = -9999.0

# =========================================================
# U-NET (SAME ARCHITECTURE AS TRAINING)
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def build_positions(full_size, patch_size, stride):
    if full_size <= patch_size:
        return [0]

    positions = list(range(0, full_size - patch_size + 1, stride))
    if positions[-1] != full_size - patch_size:
        positions.append(full_size - patch_size)
    return positions


def read_patch(src, row, col, patch_size):
    h = src.height
    w = src.width

    win_h = min(patch_size, h - row)
    win_w = min(patch_size, w - col)

    window = Window(col_off=col, row_off=row, width=win_w, height=win_h)
    patch = src.read(window=window).astype(np.float32)  # (bands, h, w)

    # Pad if we are at the raster edge
    if win_h < patch_size or win_w < patch_size:
        padded = np.zeros((patch.shape[0], patch_size, patch_size), dtype=np.float32)
        padded[:, :win_h, :win_w] = patch
        patch = padded

    return patch, win_h, win_w


def normalize_patch(patch):
    patch = patch.copy()
    for c in range(patch.shape[0]):
        band = patch[c]
        valid = np.isfinite(band)

        if np.any(valid):
            p2 = np.percentile(band[valid], 2)
            p98 = np.percentile(band[valid], 98)

            if p98 > p2:
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2)
            else:
                band = np.zeros_like(band)
        else:
            band = np.zeros_like(band)

        patch[c] = band

    patch[~np.isfinite(patch)] = 0
    return patch


def get_valid_mask(patch):
    # A pixel is valid if it has signal in at least one band
    valid = np.any(np.isfinite(patch), axis=0) & (np.sum(np.abs(patch), axis=0) != 0)
    return valid.astype(np.uint8)


def classify_density(percent_array):
    out = np.zeros(percent_array.shape, dtype=np.uint8)

    mask = ~np.isnan(percent_array)

    out[(percent_array >= 0) & (percent_array < 5) & mask] = 1
    out[(percent_array >= 5) & (percent_array < 15) & mask] = 2
    out[(percent_array >= 15) & (percent_array < 30) & mask] = 3
    out[(percent_array >= 30) & mask] = 4

    return out


def block_mean(arr, block_h, block_w):
    h, w = arr.shape
    out_h = math.ceil(h / block_h)
    out_w = math.ceil(w / block_w)

    out = np.full((out_h, out_w), np.nan, dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            r0 = i * block_h
            r1 = min((i + 1) * block_h, h)
            c0 = j * block_w
            c1 = min((j + 1) * block_w, w)

            block = arr[r0:r1, c0:c1]
            valid = ~np.isnan(block)

            if np.any(valid):
                out[i, j] = np.mean(block[valid])

    return out

# =========================================================
# STEP 1: FULL-FIELD INFERENCE
# =========================================================

def run_inference():
    print("========== STEP 1: FULL-FIELD INFERENCE ==========")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Input raster: {INPUT_RASTER}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Stride: {STRIDE}")

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with rasterio.open(INPUT_RASTER) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

        print(f"Raster size: {height} x {width}")
        print(f"Bands: {src.count}")

        row_positions = build_positions(height, PATCH_SIZE, STRIDE)
        col_positions = build_positions(width, PATCH_SIZE, STRIDE)
        total_tiles = len(row_positions) * len(col_positions)

        sum_probs = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        valid_global = np.zeros((height, width), dtype=np.uint8)

        tile_count = 0

        with torch.no_grad():
            for row in row_positions:
                for col in col_positions:
                    patch, win_h, win_w = read_patch(src, row, col, PATCH_SIZE)
                    patch = patch[:IN_CHANNELS]
                    patch = normalize_patch(patch)

                    valid_patch = get_valid_mask(patch)

                    x = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (3, 256, 256)
                    probs = probs[:, :win_h, :win_w]

                    valid_crop = valid_patch[:win_h, :win_w]

                    for c in range(NUM_CLASSES):
                        sum_probs[c, row:row+win_h, col:col+win_w] += probs[c] * valid_crop

                    count_map[row:row+win_h, col:col+win_w] += valid_crop
                    valid_global[row:row+win_h, col:col+win_w] = np.maximum(
                        valid_global[row:row+win_h, col:col+win_w],
                        valid_crop
                    )

                    tile_count += 1
                    if tile_count % 50 == 0 or tile_count == total_tiles:
                        print(f"{tile_count}/{total_tiles} tiles processed")

    count_safe = np.where(count_map == 0, 1, count_map)
    avg_probs = sum_probs / count_safe[np.newaxis, :, :]

    # 0 crop, 1 weed, 2 soil -> remap to 1, 2, 3
    pred_idx = np.argmax(avg_probs, axis=0)
    pred_class = pred_idx + 1
    pred_class[valid_global == 0] = 0

    prob_weed = avg_probs[1]
    prob_weed[valid_global == 0] = NODATA_PROB

    # Save class raster
    profile_class = profile.copy()
    profile_class.update(
        count=1,
        dtype=rasterio.uint8,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(OUT_CLASS, "w", **profile_class) as dst:
        dst.write(pred_class.astype(np.uint8), 1)

    # Save weed probability raster
    profile_prob = profile.copy()
    profile_prob.update(
        count=1,
        dtype=rasterio.float32,
        nodata=NODATA_PROB,
        compress="lzw"
    )

    with rasterio.open(OUT_PROB, "w", **profile_prob) as dst:
        dst.write(prob_weed.astype(np.float32), 1)

    info = {
        "model_path": MODEL_PATH,
        "input_raster": INPUT_RASTER,
        "out_class": OUT_CLASS,
        "out_prob_weed": OUT_PROB,
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "in_channels": IN_CHANNELS,
        "device": DEVICE
    }

    with open(OUT_INFO_INF, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Inference finished.")
    print(f"Class raster: {OUT_CLASS}")
    print(f"Weed probability raster: {OUT_PROB}")
    print(f"Inference info: {OUT_INFO_INF}")

# =========================================================
# STEP 2: DENSITY MAP
# =========================================================

def run_density():
    print("\n========== STEP 2: DENSITY MAP ==========")

    with rasterio.open(OUT_PROB) as src_prob, rasterio.open(OUT_CLASS) as src_class:
        prob = src_prob.read(1).astype(np.float32)
        pred_class = src_class.read(1).astype(np.uint8)

        transform = src_prob.transform
        crs = src_prob.crs
        profile = src_prob.profile.copy()

        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)

        block_w = max(1, round(CELL_SIZE_METERS / pixel_size_x))
        block_h = max(1, round(CELL_SIZE_METERS / pixel_size_y))

        print(f"Pixel size X: {pixel_size_x:.4f} m")
        print(f"Pixel size Y: {pixel_size_y:.4f} m")
        print(f"Density block: {block_h} x {block_w} pixels")

        # Continuous density from weed probability
        prob_valid = prob.copy()
        prob_valid[prob_valid == NODATA_PROB] = np.nan
        dens_prob = block_mean(prob_valid, block_h, block_w) * 100.0

        # Binary density from final predicted class (weed = 2)
        weed_bin = np.where(pred_class == 2, 1.0, 0.0).astype(np.float32)
        weed_bin[pred_class == 0] = np.nan
        dens_bin = block_mean(weed_bin, block_h, block_w) * 100.0

        # Reclassification into density levels
        dens_class = classify_density(dens_bin)

        # New affine transform for the aggregated grid
        new_transform = Affine(
            transform.a * block_w,
            transform.b,
            transform.c,
            transform.d,
            transform.e * block_h,
            transform.f
        )

        out_h, out_w = dens_prob.shape

        # Save probability-based density
        profile_prob = profile.copy()
        profile_prob.update(
            height=out_h,
            width=out_w,
            count=1,
            dtype=rasterio.float32,
            transform=new_transform,
            nodata=NODATA_PROB,
            compress="lzw"
        )

        dens_prob_out = dens_prob.copy()
        dens_prob_out[np.isnan(dens_prob_out)] = NODATA_PROB

        with rasterio.open(OUT_DENS_PROB, "w", **profile_prob) as dst:
            dst.write(dens_prob_out.astype(np.float32), 1)

        # Save binary density
        dens_bin_out = dens_bin.copy()
        dens_bin_out[np.isnan(dens_bin_out)] = NODATA_PROB

        with rasterio.open(OUT_DENS_BIN, "w", **profile_prob) as dst:
            dst.write(dens_bin_out.astype(np.float32), 1)

        # Save reclassified density
        profile_class = profile.copy()
        profile_class.update(
            height=out_h,
            width=out_w,
            count=1,
            dtype=rasterio.uint8,
            transform=new_transform,
            nodata=0,
            compress="lzw"
        )

        with rasterio.open(OUT_DENS_CLASS, "w", **profile_class) as dst:
            dst.write(dens_class.astype(np.uint8), 1)

        info = {
            "input_prob": OUT_PROB,
            "input_class": OUT_CLASS,
            "cell_size_meters": CELL_SIZE_METERS,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
            "block_h_pixels": block_h,
            "block_w_pixels": block_w,
            "out_dens_prob": OUT_DENS_PROB,
            "out_dens_bin": OUT_DENS_BIN,
            "out_dens_class": OUT_DENS_CLASS,
            "classes_density": {
                "1": "very_low_0_5",
                "2": "low_5_15",
                "3": "medium_15_30",
                "4": "high_over_30"
            }
        }

        with open(OUT_INFO_DENS, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    print("Density map finished.")
    print(f"Probability density: {OUT_DENS_PROB}")
    print(f"Binary density: {OUT_DENS_BIN}")
    print(f"Classified density: {OUT_DENS_CLASS}")
    print(f"Density info: {OUT_INFO_DENS}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_inference()
    run_density()
    print("\nFull process finished.")
