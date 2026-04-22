# Week 16 — Perspective Transform and SIFT

## Overview

Geometric image transformations and feature-based matching. Covers perspective (homography) warping applied to video frames for bird's-eye-view generation, and SIFT keypoint detection with Lowe's ratio test for matching features across images of the same scene.

## Contents

| Path | Description |
|------|-------------|
| `perspective-transform-and-sift.py` | Perspective warp of road video frames (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`), and SIFT keypoint detection/matching between two church images using brute-force matcher with Lowe's ratio test |
| `rural.mp4` | Input video — rural road footage used for perspective transform |
| `church1.jpeg`, `church2.jpeg` | Input image pair used for SIFT feature matching |

## Key Concepts

- Perspective transformation — 3×3 homography matrix, 4-point correspondence
- Bird's-eye-view generation — mapping oblique camera angle to top-down view
- Video frame processing — per-frame warping with `cv2.VideoCapture`
- SIFT (Scale-Invariant Feature Transform) — keypoint detection and 128-D descriptors
- Brute-force matching — `cv2.BFMatcher` with L2 distance and k-NN
- Lowe's ratio test — filtering ambiguous matches (ratio threshold 0.75)

## How to Run

```bash
# Requires OpenCV (with SIFT — opencv-contrib-python or opencv-python ≥ 4.4)
pip install opencv-python numpy

python perspective-transform-and-sift.py
# Press 'q' to exit the video window
```
