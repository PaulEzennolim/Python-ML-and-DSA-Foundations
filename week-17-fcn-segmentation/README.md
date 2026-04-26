# Week 17 — FCN Semantic Segmentation

## Overview

Fully Convolutional Networks for semantic segmentation. Compares pre-trained `fcn_resnet50` and `fcn_resnet101` on PASCAL VOC, then trains an FCN with an AlexNet backbone (with skip connections) on the JSRT chest radiograph dataset for clavicle, lung, and heart segmentation.

## Contents

| Path | Description |
|------|-------------|
| `notebooks/01_pretrained_voc_benchmark.ipynb` | Pre-trained FCN inference on natural images, ResNet-50 vs ResNet-101 comparison with the PASCAL VOC palette overlay |
| `notebooks/02_jsrt_anatomical_segmentation.ipynb` | End-to-end FCN-AlexNet training on JSRT chest X-rays, IoU and Dice evaluation, per-epoch visualisations |
| `dog1.jpg`, `sample_01.jpg`–`sample_05.jpg` | Test images for the VOC benchmark notebook |

## Key Concepts

- Fully Convolutional Networks — coarse feature maps + bilinear upsampling for dense prediction
- Skip connections — FCN-8s-style fusion of mid-level (192-ch) and deep (256-ch) features via `torch.cat`
- Spatial reconstruction — `F.interpolate` to lift backbone outputs back to input resolution
- Segmentation metrics — Intersection over Union (Jaccard) and Sørensen–Dice (F1)
- Transfer learning — ImageNet-pretrained AlexNet backbone adapted to grayscale medical images

## How to Run

```bash
pip install -r ../requirements.txt   # consolidated at repo root

jupyter notebook 01-pretrained-voc-benchmark.ipynb
jupyter notebook 02-jsrt-anatomical-segmentation.ipynb
```

The JSRT notebook downloads the dataset on first run and is designed for Google Colab with a GPU runtime.
