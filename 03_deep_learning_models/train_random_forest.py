import os
import glob
import random
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
import joblib

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.expanduser("~/TFM_WEEDS")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "data/patches/barley_v1_rgb_nir_re_indices/images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v1_rgb_nir_re_indices/masks")

TEST_IMG_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/masks")

OUT_DIR = os.path.join(BASE_DIR, "results/random_forest_rgb_nir_re_indices")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_TREES = 200
MAX_SAMPLES_TRAIN = 300000   # adjust up or down as needed
MAX_SAMPLES_TEST = 300000    # to evaluate without exceeding memory

random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# FUNCTIONS
# =========================================================

def load_pixels_from_patches(img_dir, mask_dir, max_samples=None):
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    assert len(img_paths) == len(mask_paths), "Mismatch between number of images and masks."
    assert len(img_paths) > 0, "No patches found."

    X_list = []
    y_list = []

    for img_path, mask_path in zip(img_paths, mask_paths):
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C,H,W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H,W)

        # Per-band, per-patch normalization
        for c in range(img.shape[0]):
            band = img[c]
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

            img[c] = band

        img[~np.isfinite(img)] = 0

        # Reshape to a pixel table
        C, H, W = img.shape
        X = img.reshape(C, -1).T          # (H*W, C)
        y = mask.reshape(-1)              # (H*W,)

        # Keep only valid classes
        valid = np.isin(y, [1, 2, 3])
        X = X[valid]
        y = y[valid]

        # Remap labels:
        # 1 -> 0 crop
        # 2 -> 1 weed
        # 3 -> 2 soil
        y_new = np.full_like(y, -1)
        y_new[y == 1] = 0
        y_new[y == 2] = 1
        y_new[y == 3] = 2
        y = y_new

        X_list.append(X)
        y_list.append(y)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # Random subsampling if needed
    if max_samples is not None and len(y_all) > max_samples:
        idx = np.random.choice(len(y_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    return X_all, y_all

def iou_per_class(y_true, y_pred, num_classes=3):
    ious = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        iou = tp / (tp + fp + fn + 1e-8)
        ious.append(iou)
    return ious

# =========================================================
# LOAD DATA
# =========================================================

print("Loading train...")
X_train, y_train = load_pixels_from_patches(
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, max_samples=MAX_SAMPLES_TRAIN
)

print("Loading test...")
X_test, y_test = load_pixels_from_patches(
    TEST_IMG_DIR, TEST_MASK_DIR, max_samples=MAX_SAMPLES_TEST
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# =========================================================
# MODEL
# =========================================================

rf = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Predicting...")
y_pred = rf.predict(X_test)

# =========================================================
# METRICS
# =========================================================

f1_crop = f1_score(y_test, y_pred, labels=[0], average="macro")
f1_weed = f1_score(y_test, y_pred, labels=[1], average="macro")
f1_soil = f1_score(y_test, y_pred, labels=[2], average="macro")

iou_crop, iou_weed, iou_soil = iou_per_class(y_test, y_pred, num_classes=3)
miou = (iou_crop + iou_weed + iou_soil) / 3

print("\nRandom Forest Results")
print(f"F1 crop: {f1_crop:.4f}")
print(f"F1 weed: {f1_weed:.4f}")
print(f"F1 soil: {f1_soil:.4f}")
print(f"IoU crop: {iou_crop:.4f}")
print(f"IoU weed: {iou_weed:.4f}")
print(f"IoU soil: {iou_soil:.4f}")
print(f"mIoU: {miou:.4f}")

# =========================================================
# SAVE
# =========================================================

joblib.dump(rf, os.path.join(OUT_DIR, "random_forest_model.joblib"))

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Train images: {TRAIN_IMG_DIR}\n")
    f.write(f"Train masks: {TRAIN_MASK_DIR}\n")
    f.write(f"Test images: {TEST_IMG_DIR}\n")
    f.write(f"Test masks: {TEST_MASK_DIR}\n")
    f.write(f"N trees: {N_TREES}\n")
    f.write(f"Train samples: {len(y_train)}\n")
    f.write(f"Test samples: {len(y_test)}\n\n")

    f.write(f"F1 crop: {f1_crop:.6f}\n")
    f.write(f"F1 weed: {f1_weed:.6f}\n")
    f.write(f"F1 soil: {f1_soil:.6f}\n")
    f.write(f"IoU crop: {iou_crop:.6f}\n")
    f.write(f"IoU weed: {iou_weed:.6f}\n")
    f.write(f"IoU soil: {iou_soil:.6f}\n")
    f.write(f"mIoU: {miou:.6f}\n")

print("\nModel saved to:", os.path.join(OUT_DIR, "random_forest_model.joblib"))
print("Summary saved to:", summary_path)
