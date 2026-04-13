# Week 05 — Graphs, Heaps, Sorting, and Data Analysis

## Overview

Introduces graph representations and traversal algorithms (BFS, DFS), heap data structures, classical sorting algorithms, and applied data-analysis scripts for pulse-signal exploration.

## Contents

| Path | Description |
|------|-------------|
| `graphs.ipynb` | Graph representations (adjacency list/matrix), BFS, DFS |
| `heaps.ipynb` | Min/max heaps — insertion, extraction, heapify |
| `sorting.ipynb` | Sorting algorithms — bubble, selection, insertion, merge, quick sort |
| `further-data-analysis/explore-pulses.py` | Exploratory analysis of pulse-signal datasets |
| `further-data-analysis/bin-pulses.py` | Binary pulse conversion and histogram analysis |

## Key Concepts

- Graphs — adjacency list vs matrix, directed/undirected, weighted edges
- BFS and DFS — level-order vs depth-first exploration, shortest path
- Heaps — priority queues, heap property, O(log n) insert/extract
- Sorting — comparison-based algorithms, time complexity analysis (O(n²) vs O(n log n))

## How to Run

```bash
# Notebooks
jupyter notebook graphs.ipynb
jupyter notebook sorting.ipynb

# Data analysis scripts (requires NumPy, Matplotlib)
python further-data-analysis/explore-pulses.py
```
