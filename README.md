# Python ML and DSA Foundations

A comprehensive, 17-week curriculum covering **Data Structures & Algorithms**, **Computer Vision**, and **Machine Learning** in Python. Each week builds on the previous, progressing from fundamental programming through advanced deep-learning architectures. The repository includes Jupyter notebooks, standalone scripts, unit tests, and real-world datasets.

---

## Table of Contents

| Week | Topic | Key Technologies |
|------|-------|-----------------|
| [01](week-01-arrays-and-python-basics/) | [Arrays and Python Basics](week-01-arrays-and-python-basics/) | Python, pytest |
| [02](week-02-hashmaps-sets-recursion/) | [Hashmaps, Sets, and Recursion](week-02-hashmaps-sets-recursion/) | Python |
| [03](week-03-linked-lists-and-programs/) | [Linked Lists and Loop-Based Programs](week-03-linked-lists-and-programs/) | Python |
| [04](week-04-Binary-Search-trees-and-time-series-analysis/) | [Binary Search, Trees, and Time-Series Analysis](week-04-Binary-Search-trees-and-time-series-analysis/) | NumPy, Matplotlib |
| [05](week-05-graphs-heaps-and-further-data-analysis/) | [Graphs, Heaps, Sorting, and Data Analysis](week-05-graphs-heaps-and-further-data-analysis/) | NumPy, Matplotlib |
| [06](week-06-problem-solving-patterns-and-practical-applications/) | [Problem-Solving Patterns and Practical Applications](week-06-problem-solving-patterns-and-practical-applications/) | Pillow |
| [07](week-07-dynamic-programming-and-animations/) | [Dynamic Programming and Shape Animations](week-07-dynamic-programming-and-animations/) | Matplotlib |
| [08](week-08-ml-projects/) | [Machine Learning Projects](week-08-ml-projects/) | PyTorch, scikit-learn, pandas |
| [09](week-09-cv-image-creation-and-color/) | [CV: Image Creation and Colour](week-09-cv-image-creation-and-color/) | OpenCV, NumPy |
| [10](week-10-cv-contrast-and-overlay/) | [CV: Contrast Enhancement and Overlay](week-10-cv-contrast-and-overlay/) | OpenCV, NumPy |
| [11](week-11-image-filtering-convolution/) | [Image Filtering and Convolution](week-11-image-filtering-convolution/) | OpenCV, NumPy |
| [12](week-12-edge-detection-and-hough-transform/) | [Edge Detection and Hough Transform](week-12-edge-detection-and-hough-transform/) | OpenCV, NumPy |
| [13](week-13-perspective-transform-and-sift/) | [Perspective Transform and SIFT](week-13-perspective-transform-and-sift/) | OpenCV, NumPy |
| [14](week-14-mnist-pytorch-classifier/) | [MNIST Digit Classifier (PyTorch)](week-14-mnist-pytorch-classifier/) | PyTorch, torchvision |
| [15](week-15-cnn-alexnet-cifar10/) | [CNN and AlexNet on CIFAR-10](week-15-cnn-alexnet-cifar10/) | PyTorch, torchvision |
| [16](week-16-yolov8-object-detection/) | [YOLOv8 Object Detection](week-16-yolov8-object-detection/) | Ultralytics |
| [17](week-17-fcn-segmentation/) | [FCN Semantic Segmentation](week-17-fcn-segmentation/) | PyTorch, torchvision, scikit-image |

---

## Curriculum Overview

### Phase 1 — Data Structures and Algorithms (Weeks 01–07)

Builds a strong foundation in classical DSA, starting from Python basics and progressing through increasingly complex data structures and algorithmic patterns.

- **Week 01** — Python fundamentals, binary representation, static arrays
- **Week 02** — Hash maps, sets, recursion, stacks, and queues
- **Week 03** — Singly and doubly linked lists, iterative programs
- **Week 04** — Binary search, binary search trees, time-series analysis
- **Week 05** — Graph traversal (BFS/DFS), heaps, sorting algorithms
- **Week 06** — Two pointers, sliding window, recursive backtracking, image processing, lexical analysis
- **Week 07** — Dynamic programming (memoisation, tabulation), OOP animation engine

### Phase 2 — Machine Learning (Week 08)

Applied ML projects bridging classical algorithms and deep learning.

- **Week 08** — Random Forest (classification and regression with scikit-learn), robust MNIST classifier (ensemble ResNet CNNs with data augmentation and TTA)

### Phase 3 — Computer Vision (Weeks 09–13)

Progressive introduction to computer vision fundamentals using OpenCV, building from pixel-level operations through feature detection to geometric transformations and descriptor matching.

- **Week 09** — Image creation, colour spaces (grayscale, BGR, RGB), pixel manipulation
- **Week 10** — Contrast enhancement (min-max normalisation), alpha blending, binary thresholding
- **Week 11** — Spatial filtering, manual convolution implementation, performance benchmarking
- **Week 12** — Sobel edge detection, Canny edge detector, Hough circle transform
- **Week 13** — Perspective (homography) transforms on video and SIFT feature matching between images

### Phase 4 — Deep Learning (Weeks 14–17)

End-to-end deep learning pipelines, from fully-connected networks through CNNs to real-time object detection and semantic segmentation.

- **Week 14** — Fully-connected neural network on MNIST (PyTorch DataLoader, training loop, checkpointing)
- **Week 15** — AlexNet on CIFAR-10 (from-scratch training and ImageNet transfer learning)
- **Week 16** — YOLOv8 object detection (pre-trained inference and custom dataset training)
- **Week 17** — Fully Convolutional Networks for semantic segmentation (pre-trained `fcn_resnet50/101` on PASCAL VOC, FCN-AlexNet with skip connections trained on JSRT chest radiographs for clavicle/lung/heart segmentation)

---

## Requirements

### Python

```bash
python --version   # Python 3.8+ recommended
```

### Core Dependencies

```bash
pip install numpy matplotlib jupyter
```

### Domain-Specific Dependencies

| Domain | Packages |
|--------|----------|
| Image processing | `Pillow` |
| Machine learning | `scikit-learn`, `pandas` |
| Computer vision | `opencv-python` |
| Deep learning | `torch`, `torchvision` |
| Object detection | `ultralytics` |
| Semantic segmentation | `torch`, `torchvision`, `scikit-image` |

### Quick Install (all dependencies)

The full set of dependencies for every week is pinned in [requirements.txt](requirements.txt), organised week-by-week:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to any `week-*/` folder and open the `.ipynb` file.

### Python Scripts

```bash
python week-03-linked-lists-and-programs/using-loops/factorial.py
```

### Unit Tests

```bash
pytest week-01-arrays-and-python-basics/getting-started/
pytest week-02-hashmaps-sets-recursion/conditionals-and-functions/
pytest week-06-problem-solving-patterns-and-practical-applications/lexical-analysis/
```

---

## Repository Structure

```
Python-ML-and-DSA-Foundations/
├── week-01-arrays-and-python-basics/
├── week-02-hashmaps-sets-recursion/
├── week-03-linked-lists-and-programs/
├── week-04-Binary-Search-trees-and-time-series-analysis/
├── week-05-graphs-heaps-and-further-data-analysis/
├── week-06-problem-solving-patterns-and-practical-applications/
├── week-07-dynamic-programming-and-animations/
├── week-08-ml-projects/
├── week-09-cv-image-creation-and-color/
├── week-10-cv-contrast-and-overlay/
├── week-11-image-filtering-convolution/
├── week-12-edge-detection-and-hough-transform/
├── week-13-perspective-transform-and-sift/
├── week-14-mnist-pytorch-classifier/
├── week-15-cnn-alexnet-cifar10/
├── week-16-yolov8-object-detection/
├── week-17-fcn-segmentation/
└── README.md
```

---

## Contributing

Contributions, improvements, and suggestions are welcome.
Fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
