import os
import glob
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.expanduser("~/TFM_WEEDS")

IMG_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/images")
MASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/masks")
ROWMASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/rowmask")

MODEL_PATH = os.path.join(BASE_DIR, "results/unet_rgb_nir_re_indices_30ep/best_model.pth")
OUT_DIR = os.path.join(BASE_DIR, "results/hybrid_unet_rowmask_v3_objects")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
SEED = 42

# =========================================================
# HYBRID V3 PARAMETERS (OBJECT-LEVEL ANALYSIS)
# =========================================================
TH_PIXEL_BASE = 0.50              # base threshold to generate candidates
MIN_OBJ_SIZE = 2                  # absolute minimum candidate object size

TH_OBJ_OUT_ROW = 0.45             # min. mean confidence if outside crop row
TH_OBJ_IN_ROW = 0.65              # min. mean confidence if inside crop row
MIN_SIZE_IN_ROW = 4               # min. object size if mostly inside crop row
ROW_DOMINANCE = 0.50              # if >50% of the object falls in the row, treat it as "in row"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# DATASET
# =========================================================

class HybridPatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir, rowmask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.rowmask_paths = sorted(glob.glob(os.path.join(rowmask_dir, "*.tif")))

        assert len(self.img_paths) == len(self.mask_paths) == len(self.rowmask_paths), \
            "Mismatch between number of images, masks, and row masks."

        img_names = [os.path.basename(x) for x in self.img_paths]
        mask_names = [os.path.basename(x) for x in self.mask_paths]
        row_names = [os.path.basename(x) for x in self.rowmask_paths]

        assert img_names == mask_names == row_names, \
            "Filenames for images, masks, and row masks do not match."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

        with rasterio.open(self.rowmask_paths[idx]) as src:
            rowmask = src.read(1).astype(np.uint8)

        # per-band, per-patch normalization
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

        # reference mask: 1->0 crop, 2->1 weed, 3->2 soil, rest=255
        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2
        mask = mask_new

        rowmask = (rowmask > 0).astype(np.uint8)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(rowmask, dtype=torch.uint8),
            os.path.basename(self.img_paths[idx])
        )

# =========================================================
# METRICS
# =========================================================

def compute_metrics(preds, targets, num_classes=3, ignore_index=255):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    f1_per_class = []
    iou_per_class = []

    for cls in range(num_classes):
        pred_c = preds == cls
        targ_c = targets == cls

        tp = np.sum(pred_c & targ_c)
        fp = np.sum(pred_c & ~targ_c)
        fn = np.sum(~pred_c & targ_c)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        f1_per_class.append(f1)
        iou_per_class.append(iou)

    return f1_per_class, iou_per_class

# =========================================================
# U-NET (SAME ARCHITECTURE AS TRAINING)
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
# HYBRID V3 — OBJECT-LEVEL RULES
# =========================================================

def remove_small_components(binary_mask, min_size=2):
    labeled, num = ndimage.label(binary_mask)
    out = np.zeros_like(binary_mask, dtype=np.uint8)

    for i in range(1, num + 1):
        comp = labeled == i
        if comp.sum() >= min_size:
            out[comp] = 1

    return out

def apply_hybrid_rules_object_based(prob_weed, pred_base, rowmask):
    """
    prob_weed: (H,W)
    pred_base: (H,W) classes 0 crop / 1 weed / 2 soil
    rowmask:   (H,W) 1 = inside crop row / 0 = outside
    """
    pred_h = pred_base.copy()

    # 1. Generate weed candidates from base probability
    cand = (prob_weed >= TH_PIXEL_BASE).astype(np.uint8)

    # 2. Remove absolute minimal noise
    cand = remove_small_components(cand, min_size=MIN_OBJ_SIZE)

    # 3. Analyze individual objects
    labeled, num = ndimage.label(cand)
    keep = np.zeros_like(cand, dtype=np.uint8)

    for i in range(1, num + 1):
        obj = labeled == i
        size = obj.sum()

        mean_prob = prob_weed[obj].mean()
        row_ratio = rowmask[obj].mean()   # % of the object inside the crop row

        in_row = row_ratio >= ROW_DOMINANCE

        if in_row:
            # Mostly inside the crop row: require higher confidence and size
            if mean_prob >= TH_OBJ_IN_ROW and size >= MIN_SIZE_IN_ROW:
                keep[obj] = 1
        else:
            # Outside the crop row: more permissive criterion
            if mean_prob >= TH_OBJ_OUT_ROW:
                keep[obj] = 1

    # 4. Build final prediction
    pred_h[pred_h == 1] = 0
    pred_h[keep == 1] = 1

    return pred_h

# =========================================================
# INFERENCE
# =========================================================

dataset = HybridPatchDataset(IMG_DIR, MASK_DIR, ROWMASK_DIR)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

model = UNet(in_channels=8, num_classes=NUM_CLASSES).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print("Device:", DEVICE)
print("Test patches:", len(dataset))
print("Model:", MODEL_PATH)

all_targets = []
all_preds_base = []
all_preds_hybrid = []

with torch.no_grad():
    for imgs, masks, rowmasks, names in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        preds_base = torch.argmax(probs, dim=1).cpu().numpy()
        prob_weed = probs[:, 1, :, :].cpu().numpy()

        masks_np = masks.numpy()
        row_np = rowmasks.numpy()

        for i in range(len(names)):
            pred_b = preds_base[i]
            pm = prob_weed[i]
            rm = row_np[i]
            gt = masks_np[i]

            pred_h = apply_hybrid_rules_object_based(pm, pred_b, rm)

            all_targets.append(gt)
            all_preds_base.append(pred_b)
            all_preds_hybrid.append(pred_h)

all_targets = np.stack(all_targets)
all_preds_base = np.stack(all_preds_base)
all_preds_hybrid = np.stack(all_preds_hybrid)

# =========================================================
# BASELINE METRICS
# =========================================================

f1_base, iou_base = compute_metrics(all_preds_base, all_targets, num_classes=3, ignore_index=255)
f1b_c, f1b_w, f1b_s = f1_base
ioub_c, ioub_w, ioub_s = iou_base
miou_base = (ioub_c + ioub_w + ioub_s) / 3

# =========================================================
# HYBRID METRICS
# =========================================================

f1_h, iou_h = compute_metrics(all_preds_hybrid, all_targets, num_classes=3, ignore_index=255)
f1h_c, f1h_w, f1h_s = f1_h
iouh_c, iouh_w, iouh_s = iou_h
miou_h = (iouh_c + iouh_w + iouh_s) / 3

# =========================================================
# SAVE SUMMARY
# =========================================================

summary_path = os.path.join(OUT_DIR, "hybrid_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== CONFIGURATION ===\n")
    f.write(f"Base model: {MODEL_PATH}\n")
    f.write(f"IMG_DIR: {IMG_DIR}\n")
    f.write(f"MASK_DIR: {MASK_DIR}\n")
    f.write(f"ROWMASK_DIR: {ROWMASK_DIR}\n")
    f.write(f"TH_PIXEL_BASE: {TH_PIXEL_BASE}\n")
    f.write(f"MIN_OBJ_SIZE: {MIN_OBJ_SIZE}\n")
    f.write(f"TH_OBJ_OUT_ROW: {TH_OBJ_OUT_ROW}\n")
    f.write(f"TH_OBJ_IN_ROW: {TH_OBJ_IN_ROW}\n")
    f.write(f"MIN_SIZE_IN_ROW: {MIN_SIZE_IN_ROW}\n")
    f.write(f"ROW_DOMINANCE: {ROW_DOMINANCE}\n\n")

    f.write("=== BASELINE RESULTS (U-NET) ===\n")
    f.write(f"F1 crop: {f1b_c:.6f}\n")
    f.write(f"F1 weed: {f1b_w:.6f}\n")
    f.write(f"F1 soil: {f1b_s:.6f}\n")
    f.write(f"IoU crop: {ioub_c:.6f}\n")
    f.write(f"IoU weed: {ioub_w:.6f}\n")
    f.write(f"IoU soil: {ioub_s:.6f}\n")
    f.write(f"mIoU: {miou_base:.6f}\n\n")

    f.write("=== HYBRID V3 RESULTS ===\n")
    f.write(f"F1 crop: {f1h_c:.6f}\n")
    f.write(f"F1 weed: {f1h_w:.6f}\n")
    f.write(f"F1 soil: {f1h_s:.6f}\n")
    f.write(f"IoU crop: {iouh_c:.6f}\n")
    f.write(f"IoU weed: {iouh_w:.6f}\n")
    f.write(f"IoU soil: {iouh_s:.6f}\n")
    f.write(f"mIoU: {miou_h:.6f}\n")

print("\n=== BASELINE RESULTS (U-NET) ===")
print(f"F1 crop: {f1b_c:.4f}")
print(f"F1 weed: {f1b_w:.4f}")
print(f"F1 soil: {f1b_s:.4f}")
print(f"IoU crop: {ioub_c:.4f}")
print(f"IoU weed: {ioub_w:.4f}")
print(f"IoU soil: {ioub_s:.4f}")
print(f"mIoU: {miou_base:.4f}")

print("\n=== HYBRID V3 RESULTS ===")
print(f"F1 crop: {f1h_c:.4f}")
print(f"F1 weed: {f1h_w:.4f}")
print(f"F1 soil: {f1h_s:.4f}")
print(f"IoU crop: {iouh_c:.4f}")
print(f"IoU weed: {iouh_w:.4f}")
print(f"IoU soil: {iouh_s:.4f}")
print(f"mIoU: {miou_h:.4f}")

print("\nSummary saved to:", summary_path)
