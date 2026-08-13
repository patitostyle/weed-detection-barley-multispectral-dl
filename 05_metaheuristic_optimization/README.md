# 05 — Metaheuristic Optimisation (PSO)

Uses Particle Swarm Optimisation (PSO) to search for training hyperparameters that maximise the U-Net's performance on the weed class, on top of the spatial-validation setup (training on Barley 1 V1+V2, evaluating on Barley 2).

## What it does

1. Computes class weights for the loss function from the training set's class frequencies (inverse square-root weighting, crop fixed at 1.0).
2. Runs PSO to search over 4 hyperparameters:
   - **Learning rate** (`lr`)
   - **Weight decay** (`wd`)
   - **Weed class weight factor** (multiplies the base weed weight)
   - **Soil class weight factor** (multiplies the base soil weight)
3. Each particle is evaluated by training a fresh U-Net for a short run (`EPOCHS_PSO` epochs) and scoring it with a fitness function combining weed F1-score and weed IoU (weighted 0.7 / 0.3).
4. Standard PSO update rules (inertia `w`, cognitive/social coefficients `c1`/`c2`) move particles toward the best personal and global positions found across iterations.
5. Saves the full search history and the best parameter combination found.

## Requirements

```
pip install torch rasterio numpy
```

## Inputs

- Training patches from **two flights combined** (e.g. Barley 1 V1 + Barley 1 V2) — pass a list of patch dataset folders in `TRAIN_DIRS`
- Evaluation patches from an independent field (e.g. Barley 2) in `TEST_DIR`
- Each folder must contain `images/` and `masks/` subfolders (output of `02_patch_generation/`)

## Outputs

Saved to `OUT_DIR`:
- `pso_history.json`: every particle's parameters and metrics, for every iteration
- `best_params.json`: the best parameter combination found (global best)
- `pso_summary.txt`: human-readable summary of the configuration, class statistics, and best result

## How to run

1. Edit the **CONFIG** section at the top of `pso_optimization.py`:
   - `TRAIN_DIRS`: list of training patch folders to combine
   - `TEST_DIR`: evaluation patch folder
   - `OUT_DIR`: where results are saved
   - Search ranges (`LR_MIN`/`LR_MAX`, etc.) and PSO settings (`N_PARTICLES`, `N_ITER`, `EPOCHS_PSO`) if you want a different search budget
2. Run:
   ```
   python pso_optimization.py
   ```

## Notes

- This is a **search stage**, not the final training run: each particle only trains for `EPOCHS_PSO` (12) epochs to keep the search affordable (6 particles × 4 iterations = 24 short training runs).
- Once the best parameters are found, they should be used in a full-length training run (e.g. 100 epochs) for the final comparison against the baseline model — this final run is not part of this script.
- In this study, the PSO-found parameters improved performance slightly during the short search but did **not** outperform the baseline model when evaluated with full training. See the main repo README for the comparison.
