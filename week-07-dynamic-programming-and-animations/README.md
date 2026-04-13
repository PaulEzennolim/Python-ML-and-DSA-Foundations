# Week 07 — Dynamic Programming and Shape Animations

## Overview

Dynamic programming fundamentals (memoisation, tabulation, optimal substructure) alongside an object-oriented shape-animation engine with collision detection and test suites.

## Contents

| Path | Description |
|------|-------------|
| `dynamic-programming.ipynb` | DP principles — Fibonacci, knapsack, longest common subsequence |
| `animating-shapes/Shapes.py` | Base `Shape` class — drawing primitives (square, circle, diamond) |
| `animating-shapes/MovingShapes.py` | `MovingShape` subclass — velocity, boundary bouncing, collision handling |
| `animating-shapes/load_shapes.py` | Factory for loading and instantiating shape configurations |
| `animating-shapes/test_one_shape.py` | Test: single shape movement and boundary reflection |
| `animating-shapes/test_multi_shapes.py` | Test: multiple shapes animated concurrently |
| `animating-shapes/test_interacting_shapes.py` | Test: shape–shape collision detection and response |

## Key Concepts

- Dynamic programming — overlapping subproblems, memoisation vs tabulation
- Classical DP problems — Fibonacci, 0/1 knapsack, LCS
- OOP design — inheritance, polymorphism, encapsulation
- 2D animation — frame-based rendering, velocity vectors, elastic collision

## How to Run

```bash
# Notebook
jupyter notebook dynamic-programming.ipynb

# Animation demos (requires Matplotlib)
python animating-shapes/test_one_shape.py
python animating-shapes/test_multi_shapes.py
python animating-shapes/test_interacting_shapes.py
```
