# Week 14 — MNIST Digit Classifier with PyTorch

## Overview

Deep learning with PyTorch — building, training, and evaluating a fully-connected neural network for handwritten digit classification on the MNIST dataset. Includes the complete data pipeline, training loop, and model checkpointing.

## Contents

| Path | Description |
|------|-------------|
| `mnist-pytorch-classifier.ipynb` | End-to-end MNIST classifier — DataLoader setup with transforms, network architecture, training loop, loss/accuracy tracking, model saving |
| `data/MNIST/raw/` | Raw MNIST dataset files (auto-downloaded by torchvision) |

## Key Concepts

- PyTorch `nn.Module` — defining fully-connected layers, forward pass
- `DataLoader` pipeline — batching, shuffling, torchvision transforms
- Training loop — forward pass, loss computation, backpropagation, optimizer step
- Model checkpointing — saving/loading network weights and optimizer state
- Evaluation — accuracy measurement on held-out test set

## How to Run

```bash
# Requires PyTorch and torchvision
pip install torch torchvision

jupyter notebook mnist-pytorch-classifier.ipynb
```
