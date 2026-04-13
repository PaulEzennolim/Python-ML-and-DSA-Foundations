# Week 08 — Machine Learning Projects

## Overview

Two applied ML projects: a robust MNIST digit classifier using an ensemble of ResNet-style CNNs (PyTorch), and Random Forest models for classification and regression tasks (scikit-learn).

## Contents

| Path | Description |
|------|-------------|
| `mnist-robust-classifier/system.py` | Ensemble of 5 ResNet-style CNNs with data augmentation and test-time augmentation |
| `mnist-robust-classifier/train.py` | Training pipeline — feature extraction, model fitting, checkpointing |
| `mnist-robust-classifier/evaluate.py` | Evaluation on noisy and masked test images |
| `mnist-robust-classifier/utils.py` | Dataset loading, model I/O, and helper utilities |
| `random-forest-algorithm/Random Forest for Classification Tasks.ipynb` | Random Forest classifier — Titanic survival prediction |
| `random-forest-algorithm/Random Forest for Regression Tasks.ipynb` | Random Forest regressor — continuous target prediction |

## Key Concepts

- Convolutional neural networks — residual connections, global average pooling
- Data augmentation — noise injection, rectangular mask occlusion
- Test-time augmentation (TTA) — pixel-shift ensembling
- Random Forest — bagging, feature importance, classification vs regression
- scikit-learn pipeline — train/test split, accuracy, classification reports

## How to Run

```bash
# MNIST robust classifier (requires PyTorch)
python mnist-robust-classifier/train.py
python mnist-robust-classifier/evaluate.py

# Random Forest notebooks (requires scikit-learn, pandas)
jupyter notebook "random-forest-algorithm/Random Forest for Classification Tasks.ipynb"
```
