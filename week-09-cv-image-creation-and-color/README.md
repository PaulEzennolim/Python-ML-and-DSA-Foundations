# Week 09 — Computer Vision: Image Creation and Colour

## Overview

Introduction to computer vision with OpenCV. Covers programmatic image creation using NumPy arrays, colour spaces (grayscale, BGR, RGB), and fundamental pixel-level operations.

## Contents

| Path | Description |
|------|-------------|
| `cv-image-creation-and-color.py` | Creating black, white, grayscale, and colour images from scratch; reading, displaying, and saving images with OpenCV; colour channel splitting and manipulation |

## Key Concepts

- NumPy array construction for image data (`np.zeros`, `np.ones`)
- Grayscale vs multi-channel (BGR/RGB) image representation
- OpenCV I/O — `cv2.imread`, `cv2.imshow`, `cv2.imwrite`
- Colour channel operations — splitting, merging, and reordering channels
- Pixel-level access and modification

## How to Run

```bash
# Requires OpenCV and NumPy
pip install opencv-python numpy matplotlib

python cv-image-creation-and-color.py
```
