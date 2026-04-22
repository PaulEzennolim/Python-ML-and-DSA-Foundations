# Week 15 — CNN and AlexNet on CIFAR-10

## Overview

Convolutional neural networks and transfer learning. Trains AlexNet from scratch on CIFAR-10, then fine-tunes a pre-trained ImageNet AlexNet for CIFAR-10 classification. Includes data loading, training, and evaluation results.

## Contents

| Path | Description |
|------|-------------|
| `cifar10-dataloader.ipynb` | CIFAR-10 dataset loading, transforms, visualisation, and DataLoader setup |
| `alexnet-model-training.ipynb` | AlexNet architecture, from-scratch training, and transfer learning with pre-trained ImageNet weights |
| `alexnet-evaluation-results.ipynb` | Training/validation loss and accuracy curves, per-class evaluation, confusion analysis |

## Key Concepts

- Convolutional neural networks — conv layers, pooling, feature maps
- AlexNet architecture — 5 conv layers, 3 fully-connected layers, ReLU, dropout
- Transfer learning — freezing pre-trained feature layers, fine-tuning the classifier head
- CIFAR-10 — 10-class colour image classification (32x32 images)
- Training diagnostics — loss/accuracy curves, overfitting detection

## How to Run

```bash
# Requires PyTorch and torchvision
pip install torch torchvision matplotlib

jupyter notebook cifar10-dataloader.ipynb
jupyter notebook alexnet-model-training.ipynb
```
