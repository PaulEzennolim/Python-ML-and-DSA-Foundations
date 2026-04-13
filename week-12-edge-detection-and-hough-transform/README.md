# Week 12 — Edge Detection and Hough Transform

## Overview

Differential operators for edge detection and the Hough transform for geometric feature detection. Covers first-order Sobel filters, Canny edge detection with threshold tuning, and Hough circle detection.

## Contents

| Path | Description |
|------|-------------|
| `edge-detection-and-hough-transform.py` | Sobel filters (Ix, Iy, gradient magnitude), Canny edge detection, Hough circle transform for detecting circular features |

## Key Concepts

- First-order differential operators — Sobel Ix, Iy partial derivatives
- Gradient magnitude — computing edge strength from partial derivatives
- Canny edge detection — non-maximum suppression, hysteresis thresholding
- Hough circle transform — accumulator space, circle parameter voting
- Threshold tuning — impact of threshold values on edge sensitivity

## How to Run

```bash
# Requires OpenCV, NumPy, and Matplotlib
pip install opencv-python numpy matplotlib

python edge-detection-and-hough-transform.py
```
