# Week 06 — Problem-Solving Patterns and Practical Applications

## Overview

Core algorithmic problem-solving patterns (two pointers, sliding window, recursive backtracking) paired with applied projects in image processing and lexical analysis.

## Contents

| Path | Description |
|------|-------------|
| `2-pointers.ipynb` | Two-pointer technique — sorted-array problems, pair sums, partitioning |
| `sliding-window.ipynb` | Sliding-window pattern — subarray sums, max/min windows |
| `recursive-backtracking.ipynb` | Backtracking — constraint satisfaction, N-Queens, permutations |
| `image-processing/method1-pixels.py` | Pixel-level image manipulation using direct array access |
| `image-processing/method2-triples.py` | RGB triple-based image transformations |
| `image-processing/che-effects.py` | Artistic image effects applied to sample images |
| `lexical-analysis/count_words.py` | Word frequency counter with stopword filtering |
| `lexical-analysis/test-count-words.py` | Unit tests for the word-counting module |

## Key Concepts

- Two-pointer technique — O(n) solutions for sorted-array problems
- Sliding window — fixed and variable-width window algorithms
- Recursive backtracking — pruning, constraint propagation, state-space search
- Pixel-based image processing (Pillow)
- Lexical analysis — tokenisation, stopword removal, frequency distributions

## How to Run

```bash
# Notebooks
jupyter notebook 2-pointers.ipynb
jupyter notebook sliding-window.ipynb

# Image processing (requires Pillow)
python image-processing/che-effects.py

# Lexical analysis
python lexical-analysis/count_words.py
pytest lexical-analysis/test-count-words.py
```
