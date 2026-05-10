# Week 19 — Hough Transforms and Planar Rectification

## Overview

Line and circle detection via the Hough transform, plus planar rectification as a preprocessing step that makes Hough's job easier. Builds up in three pieces: standard and probabilistic Hough line detection (`cv2.HoughLines` / `cv2.HoughLinesP`) on synthetic and natural images, Hough circle detection (`cv2.HoughCircles`) on a synthetic image and on `data.coins()`, and a mini-project that rectifies a planar object via `cv2.getPerspectiveTransform` + `cv2.warpPerspective` and compares Hough line detection before and after rectification.

## Contents

| Path | Description |
|------|-------------|
| `hough-transform.ipynb` | End-to-end notebook: Canny + `HoughLines` / `HoughLinesP` on synthetic and `data.camera()` images, `HoughCircles` on synthetic circles and on `data.coins()`, and a rectification mini-project comparing line detection before and after warping. Outputs (detection counts and overlay images) are pre-rendered for direct viewing on GitHub. |

## Key Concepts

- Hough line transform — voting in `(rho, theta)` parameter space where `rho = x·cos(theta) + y·sin(theta)`
- Standard vs probabilistic Hough — `cv2.HoughLines` returns full lines; `cv2.HoughLinesP` returns finite segments with `minLineLength` / `maxLineGap` control
- Canny pre-processing — edge maps as input to both line and circle Hough variants; tuning the `(low, high)` thresholds trades sensitivity for noise
- Hough circle transform — three-parameter `(cx, cy, r)` accumulator using the gradient method; `param2` is the accumulator-threshold strictness knob, and a tight `(minRadius, maxRadius)` bracket suppresses spurious detections
- `minDist` and non-maximum suppression — preventing duplicate detections of a single physical circle
- Planar rectification — `cv2.getPerspectiveTransform` from four corner correspondences + `cv2.warpPerspective` to map a perspective view back to a front-facing rectangle
- Why rectification helps Hough — perspective-fanned parallel edges scatter votes across many `theta` bins; rectified edges concentrate votes into sharper peaks (typically at 0° and 90°)

## How to Run

```bash
pip install -r ../requirements.txt   # consolidated at repo root

jupyter notebook hough-transform.ipynb
```
