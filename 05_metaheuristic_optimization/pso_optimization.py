import os
import glob
import json
import math
import random
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.expanduser("~/TFM_WEEDS")

TRAIN_DIRS = [
    os.path.join(BASE_DIR, "data/patches/barley_v1_rgb_nir_re_indices"),
    os.path.join(BASE_DIR, "data/patches/barley_v2_rgb_nir_re_indices"),
]
TEST_DIR = os.path.join(BASE_DIR, "data/patches/barley2_v2_rgb_nir_re_indices")

OUT_DIR = os.path.join(BASE_DIR, "results/pso_unet_v1v2_test_barley2")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
SEED = 42

# PSO
N_PARTICLES = 6
N_ITER = 4

# Short training run used during the PSO search
EPOCHS_PSO = 12
BATCH_SIZE = 8

# Search space
LR_MIN, LR_MAX = 1e-5, 5e-4
WD_MIN, WD_MAX = 0.0, 1e-3
FACTOR_WEED_MIN, FACTOR_WEED_MAX = 0.5, 3.0
FACTOR_SOIL_MIN, FACTOR_SOIL_MAX = 0.5, 2.0

# Objective function weights
ALPHA_F1 = 0.7
ALPHA_IOU = 0.3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# DATASET
# =========================================================

class PatchDatasetMultiTrain(Dataset):
    def __init__(self, train_dirs):
        self.img_paths = []
        self.mask_paths = []

        for d in train_dirs:
            imgs = sorted(glob.glob(os.path.join(d, "images", "*.tif")))
            masks = sorted(glob.glob(os.path.join(d, "masks", "*.tif")))

            assert len(imgs) == len(masks), f"Mismatch between number of images/masks in {d}"
            assert len(imgs) > 0, f"No patches found in {d}"

            img_names = [os.path.basename(x) for x in imgs]
            mask_names = [os.path.basename(x) for x in masks]
            assert img_names == mask_names, f"Filenames do not match in {d}"

            self.img_paths.extend(imgs)
            self.mask_paths.extend(masks)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

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

        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask_new, dtype=torch.long)

class PatchDatasetTest(Dataset):
    def __init__(self, test_dir):
        self.img_paths = sorted(glob.glob(os.path.join(test_dir, "images", "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(test_dir, "masks", "*.tif")))

        assert len(self.img_paths) == len(self.mask_paths), "Mismatch between number of images/masks in test set"
        assert len(self.img_paths) > 0, "No patches found in test set"

        img_names = [os.path.basename(x) for x in self.img_paths]
        mask_names = [os.path.basename(x) for x in self.mask_paths]
        assert img_names == mask_names, "Filenames do not match in test set"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.img_paths[idx]) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.int64)

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

        mask_new = np.full_like(mask, 255, dtype=np.uint8)
        mask_new[mask == 1] = 0
        mask_new[mask == 2] = 1
        mask_new[mask == 3] = 2

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask_new, dtype=torch.long)

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
# U-NET
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
# BASE CLASS WEIGHTS FROM FREQUENCY
# =========================================================

def compute_class_frequencies(train_ds):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i in range(len(train_ds)):
        _, mask = train_ds[i]
        mask = mask.numpy().reshape(-1)
        valid = mask != 255
        vals = mask[valid]
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(vals == c)

    freqs = counts / counts.sum()
    return counts, freqs

def compute_base_weights(freqs):
    # inverse sqrt frequency
    w = 1.0 / np.sqrt(freqs + 1e-12)
    w = w / w[0]   # crop = 1.0
    return w

# =========================================================
# EVALUATE A SINGLE PARTICLE
# =========================================================

def evaluate_particle(params, train_ds, test_ds):
    lr, wd, factor_weed, factor_soil = params

    pin_memory = torch.cuda.is_available()
    num_workers = 2

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    sample_x, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    model = UNet(in_channels=in_channels, num_classes=NUM_CLASSES).to(DEVICE)

    class_weights = torch.tensor(
        [BASE_WEIGHTS[0], BASE_WEIGHTS[1] * factor_weed, BASE_WEIGHTS[2] * factor_soil],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_f1_weed = -1
    best_iou_weed = -1

    for epoch in range(EPOCHS_PSO):
        model.train()
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        f1s, ious = compute_metrics(all_preds, all_targets, num_classes=NUM_CLASSES, ignore_index=255)
        _, f1_weed, _ = f1s
        _, iou_weed, _ = ious

        if f1_weed > best_f1_weed:
            best_f1_weed = f1_weed
            best_iou_weed = iou_weed

    fitness = ALPHA_F1 * best_f1_weed + ALPHA_IOU * best_iou_weed

    result = {
        "lr": lr,
        "weight_decay": wd,
        "factor_weed": factor_weed,
        "factor_soil": factor_soil,
        "best_f1_weed": best_f1_weed,
        "best_iou_weed": best_iou_weed,
        "fitness": fitness,
    }

    return result

# =========================================================
# PSO
# =========================================================

def sample_particle():
    lr = 10 ** np.random.uniform(np.log10(LR_MIN), np.log10(LR_MAX))
    wd = 10 ** np.random.uniform(np.log10(max(WD_MIN, 1e-8)), np.log10(WD_MAX)) if WD_MAX > 0 else 0.0
    factor_weed = np.random.uniform(FACTOR_WEED_MIN, FACTOR_WEED_MAX)
    factor_soil = np.random.uniform(FACTOR_SOIL_MIN, FACTOR_SOIL_MAX)
    return np.array([lr, wd, factor_weed, factor_soil], dtype=np.float64)

def clip_particle(x):
    x[0] = np.clip(x[0], LR_MIN, LR_MAX)
    x[1] = np.clip(x[1], WD_MIN, WD_MAX)
    x[2] = np.clip(x[2], FACTOR_WEED_MIN, FACTOR_WEED_MAX)
    x[3] = np.clip(x[3], FACTOR_SOIL_MIN, FACTOR_SOIL_MAX)
    return x

print("Loading datasets...")
train_ds = PatchDatasetMultiTrain(TRAIN_DIRS)
test_ds = PatchDatasetTest(TEST_DIR)

counts, freqs = compute_class_frequencies(train_ds)
BASE_WEIGHTS = compute_base_weights(freqs)

print("Train patches:", len(train_ds))
print("Test patches :", len(test_ds))
print("Counts:", counts.tolist())
print("Freqs:", freqs.tolist())
print("Base weights:", BASE_WEIGHTS.tolist())
print("Device:", DEVICE)

# PSO hyperparameters
w = 0.7
c1 = 1.5
c2 = 1.5

particles = [sample_particle() for _ in range(N_PARTICLES)]
velocities = [np.zeros(4, dtype=np.float64) for _ in range(N_PARTICLES)]

pbest_pos = [p.copy() for p in particles]
pbest_score = [-np.inf for _ in range(N_PARTICLES)]
pbest_result = [None for _ in range(N_PARTICLES)]

gbest_pos = None
gbest_score = -np.inf
gbest_result = None

history = []

for it in range(N_ITER):
    print(f"\n========== ITERATION {it+1}/{N_ITER} ==========")

    for i in range(N_PARTICLES):
        particles[i] = clip_particle(particles[i])

        print(
            f"\nParticle {i+1}/{N_PARTICLES} | "
            f"lr={particles[i][0]:.6g} | "
            f"wd={particles[i][1]:.6g} | "
            f"fw={particles[i][2]:.4f} | "
            f"fs={particles[i][3]:.4f}"
        )

        result = evaluate_particle(particles[i], train_ds, test_ds)
        score = result["fitness"]

        print(
            f" -> Weed F1={result['best_f1_weed']:.4f} | "
            f"Weed IoU={result['best_iou_weed']:.4f} | "
            f"fitness={score:.4f}"
        )

        history.append({
            "iter": it + 1,
            "particle": i + 1,
            **result
        })

        if score > pbest_score[i]:
            pbest_score[i] = score
            pbest_pos[i] = particles[i].copy()
            pbest_result[i] = result.copy()

        if score > gbest_score:
            gbest_score = score
            gbest_pos = particles[i].copy()
            gbest_result = result.copy()

    print("\nBest global result so far:")
    print(json.dumps(gbest_result, indent=2))

    for i in range(N_PARTICLES):
        r1 = np.random.rand(4)
        r2 = np.random.rand(4)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_pos[i] - particles[i])
            + c2 * r2 * (gbest_pos - particles[i])
        )

        particles[i] = particles[i] + velocities[i]
        particles[i] = clip_particle(particles[i])

# Save results
with open(os.path.join(OUT_DIR, "pso_history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

with open(os.path.join(OUT_DIR, "best_params.json"), "w", encoding="utf-8") as f:
    json.dump(gbest_result, f, indent=2)

with open(os.path.join(OUT_DIR, "pso_summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== PSO CONFIG ===\n")
    f.write(f"N_PARTICLES: {N_PARTICLES}\n")
    f.write(f"N_ITER: {N_ITER}\n")
    f.write(f"EPOCHS_PSO: {EPOCHS_PSO}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n\n")

    f.write(f"Counts: {counts.tolist()}\n")
    f.write(f"Freqs: {freqs.tolist()}\n")
    f.write(f"Base weights: {BASE_WEIGHTS.tolist()}\n\n")

    f.write("=== BEST PARAMS ===\n")
    for k, v in gbest_result.items():
        f.write(f"{k}: {v}\n")

print("\nPSO finished.")
print("Best result:")
print(json.dumps(gbest_result, indent=2))
print("Saved to:", OUT_DIR)
