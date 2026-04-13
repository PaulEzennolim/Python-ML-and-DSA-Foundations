# Week 10 — Computer Vision: Contrast Enhancement and Overlay

## Overview

Image contrast enhancement and compositing techniques using OpenCV. Covers intensity statistics, min-max normalisation, alpha blending for image overlays, and binary thresholding.

## Contents

| Path | Description |
|------|-------------|
| `cv-contrast-and-overlay.py` | Intensity statistics (min, max, mean), min-max normalisation, image overlay with alpha blending, binary thresholding on fingerprint images |

## Key Concepts

- Intensity statistics — computing min, max, and mean pixel values
- Min-max normalisation — stretching contrast to the full [0, 255] range
- Image overlay — alpha blending and logo/advert compositing
- Binary thresholding — converting grayscale images to black-and-white masks
- Application to real images (dark scenes, fingerprint enhancement)

## How to Run

```bash
# Requires OpenCV and NumPy
pip install opencv-python numpy matplotlib

python cv-contrast-and-overlay.py
```
