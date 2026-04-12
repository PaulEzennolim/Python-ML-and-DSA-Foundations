# Python-DSA-Foundations

## Project Overview

This repository contains a comprehensive collection of Python exercises, scripts, and Jupyter notebooks organized into weekly themes. Each week explores a major Data Structures & Algorithms (DSA) topic, progressing from fundamental programming concepts to advanced problem-solving techniques.  
In addition to core DSA materials, the project includes mini-programs, data-analysis tools, image-processing scripts, dynamic animations, and machine-learning examples.

## Project Structure

### **week-01-arrays-and-python-basics**
Foundational Python concepts, binary representation, introductory algorithms, and basic programs.

Includes:
- Getting-started scripts (input handling, converters, number checkers)
- Intro notebooks (binary, static arrays, general Python review)
- Simple standalone programs

---

### **week-02-hashmaps-sets-recursion**
Covers hashmaps, sets, recursion, stacks, queues, and practical conditional-logic programs.

Includes:
- Conditional and function-based programs
- Hashmap/set notebooks
- Recursion and stack/queue notebooks
- SimpleCashPoint project with tests

---

### **week-03-linked-lists-and-programs**
Implementation and operations of singly and doubly linked lists, plus programs using iterative loops.

Includes:
- Loop-based programs (factorial, guessing game, Newton's method)
- Linked list data-structure notebooks

---

### **week-04-Binary-Search-trees-and-time-series-analysis**
Binary search patterns, tree structures, and introductory data-analysis scripts.

Includes:
- Tree notebooks and binary search tutorials
- Time-series analysis folder:
  - Signal generation and plotting
  - Moving-average filtering
  - Noise datasets and processing tools

---

### **week-05-graphs-heaps-and-further-data-analysis**
Graph algorithms and heap structures with additional data-analysis utilities.

Includes:
- BFS/DFS graph exploration notebooks
- Heap operations
- Sorting algorithms
- Further data analysis folder:
  - Pulse-data exploration
  - Binary pulse conversion
  - Graph plotting scripts

---

### **week-06-problem-solving-patterns-and-practical-applications**
Core algorithmic patterns and applied programming exercises.

Includes:
- Two-pointers techniques
- Sliding-window patterns
- Recursive backtracking
- Image-processing folder:
  - Pixel-based transformations
  - Image effects (with PNG sample images)
- Lexical analysis folder:
  - Word counting programs
  - Stopword filtering
  - Large text samples (Moby Dick, George texts)

---

### **week-07-dynamic-programming-animation-and-machine-learning**
Dynamic-programming fundamentals and advanced applied programs.

Includes:
- Dynamic programming notebook
- Shape animation project:
  - Shape loading, movement simulation, multi-shape interaction
  - Test suites for the animation engine
- Random-forest algorithms:
  - Classification and regression notebooks
  - Titanic dataset experiments

---

### **week-08-cv-image-creation-and-color**
Introduction to computer vision with OpenCV — creating and manipulating images programmatically.

Includes:
- Creating black, white, grayscale, and color (BGR/RGB) images from scratch using NumPy arrays
- Reading, displaying, and saving images with OpenCV
- Color channel operations and basic pixel manipulation

---

### **week-09-cv-contrast-and-overlay**
Image contrast enhancement and compositing techniques.

Includes:
- Computing intensity statistics (min, max, mean) on grayscale images
- Min-max normalization to enhance contrast
- Image overlay and alpha blending (logo/advert compositing)
- Binary thresholding applied to fingerprint images

---

### **week-10-image-filtering-convolution**
Spatial filtering, cross-correlation, and convolution from first principles.

Includes:
- Noise filtering on real images (corridor scene)
- Manual cross-correlation and 2D convolution implementations
- Slow vs fast convolution comparison against OpenCV's `cv2.filter2D`
- Padding-based filtering to preserve image borders

---

### **week-11-edge-detection-and-hough-transform**
Differential operators, edge detection, and circle detection.

Includes:
- First-order differential operations using Sobel filters (Ix, Iy, gradient magnitude)
- Canny edge detection with threshold tuning
- Hough circle transform for detecting circular features in images

---

### **week-12-mnist-pytorch-classifier**
Deep learning with PyTorch — classifying handwritten digits.

Includes:
- Fully-connected neural network trained on the MNIST dataset
- PyTorch DataLoader pipeline with transforms
- Model checkpointing (network weights and optimizer state)

---

### **week-13-cnn-alexnet-cifar10**
Convolutional neural networks and transfer learning on CIFAR-10.

Includes:
- AlexNet trained from scratch on CIFAR-10
- Transfer learning: fine-tuning a pre-trained AlexNet (ImageNet) for CIFAR-10 classification
- CIFAR-10 DataLoader setup, training loop, and accuracy/loss evaluation
- Evaluation results notebook

---

### **week-14-yolov8-object-detection**
Real-time object detection using YOLOv8 (Ultralytics).

Includes:
- Object detection with pre-trained YOLOv8 Nano and Extra-Large models
- Bounding box extraction, class labels, and confidence scores
- Custom datasets (Roboflow format) for two projects:
  - **Aerial Cars** — detecting vehicles from aerial imagery
  - **CanCounter** — counting cans in images
- Training runs and model weights

---

## Requirements

Ensure Python is installed:

```bash
python --version
```

Recommended:
- Jupyter Notebook
- Spyder/VS Code
- NumPy, Matplotlib, Pillow
- scikit-learn
- OpenCV (`cv2`)
- PyTorch (`torch`, `torchvision`)
- Ultralytics (`ultralytics`) for YOLOv8

## How to run
Navigate to the desired directory and run any notebook or script:

```bash
jupyter notebook <file_name>.ipynb
```

or

```bash
python <file_name>.py
```

Example:

```bash
python week-03/using-loops/factorial.py
```

## Contributing
Contributions, improvements, or suggestions are welcomed.
Fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License – see the LICENSE file for details.