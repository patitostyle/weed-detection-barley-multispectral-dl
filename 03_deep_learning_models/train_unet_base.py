import os
import glob
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.expanduser("~/TFM_WEEDS")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "data/patches/barley_v1_rgb_nir_re_indices/images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v1_rgb_nir_re_indices/masks")

TEST_IMG_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices/masks")

OUT_DIR = os.path.join(BASE_DIR, "results/unet_rgb_nir_re_indices_30ep")

os.makedirs(OUT_DIR, exist_ok=True)

# Training
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
SEED = 42
NUM_CLASSES = 3   # crop, weed, soil
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# DATASET
# =========================================================

class PatchDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

        assert len(self.img_paths) == len(self.mask_paths), (
            f"Mismatch between number of images and masks.\n"
            f"Images: {len(self.img_paths)} in {img_dir}\n"
            f"Masks: {len(self.mask_paths)} in {mask_dir}"
        )

        assert len(self.img_paths) > 0, (
            f"No patches found in:\n{img_dir}"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (H, W)

        # Simple per-patch, per-band normalization
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

        # Remap mask labels:
        # 0 -> 255 ignore
        # 1 -> 0 crop
        # 2 -> 1 weed
        # 3 -> 2 soil
        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2
        mask = mask_new

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

# =========================================================
# SIMPLE U-NET
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

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
# METRICS
# =========================================================

def compute_metrics(preds, targets, num_classes=3, ignore_index=255):
    preds = preds.view(-1)
    targets = targets.view(-1)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    f1_per_class = []
    iou_per_class = []

    for cls in range(num_classes):
        pred_c = preds == cls
        targ_c = targets == cls

        tp = (pred_c & targ_c).sum().item()
        fp = (pred_c & ~targ_c).sum().item()
        fn = (~pred_c & targ_c).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        f1_per_class.append(f1)
        iou_per_class.append(iou)

    return f1_per_class, iou_per_class

# =========================================================
# DATALOADERS
# =========================================================

print("Paths used:")
print("Train images:", TRAIN_IMG_DIR)
print("Train masks :", TRAIN_MASK_DIR)
print("Test images :", TEST_IMG_DIR)
print("Test masks  :", TEST_MASK_DIR)
print("Output dir  :", OUT_DIR)

train_ds = PatchDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
test_ds = PatchDataset(TEST_IMG_DIR, TEST_MASK_DIR)

# pin_memory=True is useful on GPU; on CPU it doesn't help and can cause issues
pin_memory = torch.cuda.is_available()

# First test on a shared cluster: 2 is safer
# If everything runs fine, you can increase to 4
num_workers = 2

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print(f"pin_memory: {pin_memory}")
print(f"num_workers: {num_workers}")

# Automatically detect number of input channels
sample_x, _ = train_ds[0]
in_channels = sample_x.shape[0]

# =========================================================
# MODEL
# =========================================================

model = UNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\nDevice: {DEVICE}")
print(f"Input channels: {in_channels}")
print(f"Train patches: {len(train_ds)}")
print(f"Test patches: {len(test_ds)}")

# =========================================================
# TRAIN
# =========================================================

best_test_f1_weed = -1
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    test_loss /= len(test_loader)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    f1s, ious = compute_metrics(
        all_preds,
        all_targets,
        num_classes=NUM_CLASSES,
        ignore_index=255
    )

    # 0 crop, 1 weed, 2 soil
    f1_crop, f1_weed, f1_soil = f1s
    iou_crop, iou_weed, iou_soil = ious
    miou = (iou_crop + iou_weed + iou_soil) / 3

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "f1_crop": f1_crop,
        "f1_weed": f1_weed,
        "f1_soil": f1_soil,
        "iou_crop": iou_crop,
        "iou_weed": iou_weed,
        "iou_soil": iou_soil,
        "miou": miou
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"F1 crop: {f1_crop:.4f} | "
        f"F1 weed: {f1_weed:.4f} | "
        f"F1 soil: {f1_soil:.4f} | "
        f"IoU crop: {iou_crop:.4f} | "
        f"IoU weed: {iou_weed:.4f} | "
        f"IoU soil: {iou_soil:.4f} | "
        f"mIoU: {miou:.4f}"
    )

    if f1_weed > best_test_f1_weed:
        best_test_f1_weed = f1_weed
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

# =========================================================
# SAVE SUMMARY
# =========================================================

summary_path = os.path.join(OUT_DIR, "summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Train images: {TRAIN_IMG_DIR}\n")
    f.write(f"Train masks: {TRAIN_MASK_DIR}\n")
    f.write(f"Test images: {TEST_IMG_DIR}\n")
    f.write(f"Test masks: {TEST_MASK_DIR}\n")
    f.write(f"Input channels: {in_channels}\n")
    f.write(f"Best weed F1: {best_test_f1_weed:.6f}\n\n")

    for h in history:
        f.write(
            f"Epoch {h['epoch']} | "
            f"train_loss={h['train_loss']:.6f} | "
            f"test_loss={h['test_loss']:.6f} | "
            f"f1_crop={h['f1_crop']:.6f} | "
            f"f1_weed={h['f1_weed']:.6f} | "
            f"f1_soil={h['f1_soil']:.6f} | "
            f"iou_crop={h['iou_crop']:.6f} | "
            f"iou_weed={h['iou_weed']:.6f} | "
            f"iou_soil={h['iou_soil']:.6f} | "
            f"miou={h['miou']:.6f}\n"
        )

print("\nTraining finished.")
print(f"Best weed F1: {best_test_f1_weed:.4f}")
print(f"Model saved to: {os.path.join(OUT_DIR, 'best_model.pth')}")
print(f"Summary saved to: {summary_path}")
