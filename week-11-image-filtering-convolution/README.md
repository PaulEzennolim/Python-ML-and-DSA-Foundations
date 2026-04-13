# Week 11 — Image Filtering and Convolution

## Overview

Spatial filtering from first principles. Implements cross-correlation and 2D convolution manually, compares performance against OpenCV's optimised `cv2.filter2D`, and explores padding strategies for border handling.

## Contents

| Path | Description |
|------|-------------|
| `image-filtering-convolution.py` | Noise filtering on corridor images, manual cross-correlation and 2D convolution, slow vs fast implementations, OpenCV comparison, padding-based border preservation |
| `week_3_cross_correlation_and_covolution_playground.py` | Interactive playground for experimenting with correlation and convolution kernels |

## Key Concepts

- Cross-correlation vs convolution — kernel flipping, mathematical definitions
- Manual 2D convolution — nested-loop implementation on NumPy arrays
- Performance benchmarking — `timeit` comparison of custom vs `cv2.filter2D`
- Noise filtering — mean and Gaussian kernels applied to noisy images
- Padding strategies — zero-padding, replicate-padding to preserve image borders

## How to Run

```bash
# Requires OpenCV, NumPy, and Matplotlib
pip install opencv-python numpy matplotlib

python image-filtering-convolution.py
```
