# Week 02 — Hashmaps, Sets, and Recursion

## Overview

Explores hash-based data structures (dictionaries, sets), recursion, and linear abstract data types (stacks and queues). Includes a practical cash-point project with unit tests.

## Contents

| Path | Description |
|------|-------------|
| `hashmaps-sets.ipynb` | Hash tables, Python dictionaries, and set operations |
| `recursion.ipynb` | Recursive thinking — base cases, call stacks, classic problems |
| `stacks-queues.ipynb` | Stack (LIFO) and queue (FIFO) implementations and applications |
| `conditionals-and-functions/SimpleCashPoint.py` | ATM cash-dispensing logic using conditionals and functions |
| `conditionals-and-functions/test_cashpoint.py` | Unit tests for the cash-point module |

## Key Concepts

- Hash maps — collision handling, O(1) average lookup
- Sets — membership testing, union, intersection, difference
- Recursion — base/recursive cases, stack frames, tail recursion
- Stacks and queues — push/pop, enqueue/dequeue, BFS vs DFS use cases

## How to Run

```bash
# Notebooks
jupyter notebook hashmaps-sets.ipynb

# Cash-point program & tests
python conditionals-and-functions/SimpleCashPoint.py
pytest conditionals-and-functions/test_cashpoint.py
```
