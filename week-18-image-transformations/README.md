# Week 18 — Image Transformations

## Overview

2D image transformations in homogeneous coordinates. Builds rotation, translation, and scaling as 3×3 matrices from scratch, verifies them against closed-form expectations in unit tests, and then demonstrates why the **order of matrix composition** changes the result — first on point sets, then on real grayscale images via `cv2.warpAffine`.

## Contents

| Path | Description |
|------|-------------|
| `image-transformations.ipynb` | End-to-end notebook: hand-built rotation/translation/scaling matrices, unit tests, square-point composition demos (`T @ R @ S` vs `R @ T @ S` vs `S @ R @ T`), and image warping with the centred-rotation pattern. Outputs are pre-rendered for direct viewing on GitHub. |

## Key Concepts

- Homogeneous coordinates — representing 2D points as `[x, y, 1]ᵀ` so translation becomes a matrix multiplication
- Hand-built primitive matrices — rotation `[[c,-s,0],[s,c,0],[0,0,1]]`, translation, and scaling as 3×3 matrices
- Matrix composition is non-commutative — `T @ R @ S`, `R @ T @ S`, and `S @ R @ T` produce visibly different shapes
- Centre-of-rotation trick — `C_back @ M @ C_to_origin` to apply transforms about the image centre instead of the pixel-(0,0) corner
- `cv2.warpAffine` — passing the first two rows of the 3×3 homogeneous matrix as a 2×3 affine matrix
- Geometric properties of scaling — non-uniform `sx ≠ sy` breaks aspect ratio (affine, not similarity); negative scale (`scaling_matrix(-1, 1)`) is a reflection

## How to Run

```bash
pip install -r ../requirements.txt   # consolidated at repo root

jupyter notebook image-transformations.ipynb
```

The notebook embeds its own outputs, so GitHub viewers can read results without executing anything. To regenerate the embedded outputs locally:

```bash
jupyter nbconvert --to notebook --execute --inplace image-transformations.ipynb
```
