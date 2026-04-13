# Week 15 — YOLOv8 Object Detection

## Overview

Real-time object detection using Ultralytics YOLOv8. Covers inference with pre-trained models (Nano and Extra-Large), bounding box extraction, and custom dataset training on two projects: aerial vehicle detection and can counting.

## Contents

| Path | Description |
|------|-------------|
| `yolov8-object-detection.ipynb` | YOLOv8 inference, bounding box extraction, class labels, confidence scores, and custom training |
| `Aerial Cars/data.yaml` | Roboflow-format dataset config for aerial vehicle detection |
| `CanCounter/data.yaml` | Roboflow-format dataset config for can-counting detection |

## Key Concepts

- YOLO architecture — single-pass detection, anchor-free prediction heads
- Pre-trained models — YOLOv8n (Nano) vs YOLOv8x (Extra-Large) trade-offs
- Bounding box format — xyxy coordinates, class IDs, confidence scores
- Custom dataset training — Roboflow YOLO format, `data.yaml` configuration
- Transfer learning for detection — fine-tuning COCO-pretrained weights on domain-specific data

## How to Run

```bash
# Requires Ultralytics
pip install ultralytics

jupyter notebook yolov8-object-detection.ipynb
```
