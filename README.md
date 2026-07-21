# Weed Detection in Barley Crops Using Multispectral Imagery and Deep Learning Models

Master's Thesis (TFM) — Máster Universitario en Agricultura de Precisión
Universidad Politécnica de Madrid (ETSIAAB), Departamento de Ingeniería Agroforestal
Author: Patricio Alonso Hernández Ludeña | Supervisor: Carlos Fernández Piñar

## Overview

Weeds compete with crops for water, nutrients, light, and space, reducing yield and increasing reliance on herbicides. This project develops and evaluates a semantic segmentation system to distinguish **crop, weeds, and soil** in barley fields using UAV-acquired multispectral imagery and deep learning models, with the goal of supporting site-specific weed management.

The system was built and validated end to end: UAV image acquisition, GIS preprocessing, vegetation index calculation, dataset construction, model training and comparison, spatial validation on an independent field, hyperparameter optimization, cross-crop transfer testing, and generation of final weed probability/density maps.

## Study Area & Data

- **Location:** Alcalá de Henares, Community of Madrid, Spain
- **Fields:** Barley 1 (11.83 ha), Barley 2 (15.93 ha), Oilseed rape (8.39 ha)
- **Flights:** V1 (3 March) — early growth stage; V2 (14 May) — heading/flowering stage
- **Sensors:** DJI Phantom 4 RTK (RGB, 20 MP) + DJI Matrice 600 Pro with MicaSense Altum-PT (Red, Green, Blue, NIR, RedEdge)

## Methodology

1. **GIS preprocessing (ArcGIS Pro):** orthomosaic clipping to study area (consistent path/row alignment across bands), band stacking, and calculation of vegetation indices (NDVI, GNDVI, NDRE, CCCI, SAVI, VARI)
2. **Correlation analysis** to select the most informative, least redundant indices → **NDVI, NDRE, VARI**
3. **Manual labelling** in ArcGIS Pro (crop / weed / soil) and rasterization into reference masks
4. **Patch generation:** 256×256 px patches with 50% overlap, preserving image–mask correspondence
5. **Input configuration comparison:** RGB / RGB+NIR+RedEdge / RGB+indices / RGB+NIR+RedEdge+indices
6. **Model comparison:** U-Net vs. DeepLabv3 vs. Random Forest
7. **Modular hybrid post-processing:** spatial/agronomic rule-based refinement of predictions
8. **Hyperparameter optimization:** Particle Swarm Optimisation (PSO) on the best-performing model
9. **Spatial validation** on an independent barley field (Barley 2)
10. **Cross-crop transfer test** on oilseed rape (no retraining)
11. **Weed probability & density map generation** via sliding-window spatial inference

## Key Results

| Stage | Weed F1 | Weed IoU |
|---|---|---|
| Best input configuration (RGB+NIR+RedEdge+indices, U-Net) | 0.5833 | 0.4117 |
| U-Net vs. DeepLabv3 vs. Random Forest | U-Net best (mIoU 0.5448) | — |
| Modular hybrid variants | ≤ baseline | ≤ baseline |
| **Spatial validation (Barley 1 → Barley 2)** | **0.6455** | **0.4766** |
| PSO-optimised model (full training) | 0.6378 | 0.4682 |
| Cross-crop transfer (barley → oilseed rape) | 0.0534 | 0.0274 |

**Main takeaways:**
- Combining multispectral bands (NIR, RedEdge) with vegetation indices (NDVI, NDRE, VARI) outperformed RGB-only input.
- U-Net clearly outperformed DeepLabv3 and Random Forest for the weed class.
- Training on two phenologically diverse flights improved spatial generalization within barley.
- The modular hybrid approach and PSO optimisation did not improve on the baseline U-Net.
- Direct cross-crop transfer to oilseed rape failed, highlighting the need for domain adaptation or fine-tuning when applying the model to a different crop.
- Final probability and density maps demonstrate practical potential for site-specific weed management.

## Repository Structure

```
weed-detection-barley-multispectral-dl/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── 01_gis_processing/          # Orthomosaic clipping (aligned path/row) + vegetation index calculation
├── 02_patch_generation/        # Python scripts to generate 256x256 patches from images + masks
├── 03_deep_learning_models/    # U-Net, DeepLabv3, Random Forest — training & evaluation
├── 04_metaheuristic_optimization/  # PSO hyperparameter optimisation
├── results/                    # Figures, metrics, probability/density maps
└── docs/
    ├── thesis_summary.md
    └── article_draft.md
```

## Tech Stack

- **Python:** data processing, patch generation, model training
- **Deep Learning:** U-Net, DeepLabv3 (custom implementations)
- **GIS:** ArcGIS Pro (clipping, band alignment, vegetation indices, manual labelling)
- **Photogrammetry:** Pix4D, Agisoft Metashape (orthomosaic generation)

## Citation

If you use this work, please cite the associated Master's Thesis (UPM, 2026) and/or the derived article draft.

## Contact

Patricio Hernández — patricio.hernandez@alumnos.upm.es
