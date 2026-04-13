#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script contains:
  - Noise filtering for corridor_noisy.png (Group I)
  - Cross-correlation + convolution for 2D patches (Group II)
  - Image filtering via convolution (slow + fast versions)
  - Comparison against OpenCV cv2.filter2D (speed + result)
  - Padding-based filtering to avoid zero borders

Author: paulezennolim
Created: Wed Feb 25 14:53:28 2026
"""

import cv2
import numpy as np
import timeit


# =============================================================================
# Load real image for Q10 comparison (grayscale => 2D array)
# =============================================================================
# NOTE: Put think_tank.jpeg in the same folder as this script, or give a full path.
IMG_PATH = "think_tank.jpeg"
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"{IMG_PATH} not found (check filename/path)")

# Convert to float so convolution uses float arithmetic (avoids uint8 truncation)
img = img.astype(np.float64)


# =============================================================================
# Noise Filtering (corridor_noisy.png)
# =============================================================================
def _estimate_noise_strength(channel: np.ndarray) -> float:
    """
    Simple noise proxy: std-dev of Laplacian (amount of high-frequency variation).
    Higher typically indicates more noise/edges.
    """
    lap = cv2.Laplacian(channel, cv2.CV_64F)
    return float(lap.std())


def _looks_like_salt_and_pepper(channel: np.ndarray, thresh: int = 3, frac: float = 0.02) -> bool:
    """
    Heuristic for salt & pepper (impulse) noise:
    - Counts fraction of pixels near 0 and near 255.
    - If the fraction is large enough => likely salt & pepper.
    """
    near_black = (channel <= thresh).mean()
    near_white = (channel >= 255 - thresh).mean()
    return (near_black + near_white) >= frac


def filter_corridor(
    input_path: str = "corridor_noisy.png",
    output_path: str = "corridor_filtered.png",
    preview: bool = False,
) -> np.ndarray:
    """
    Remove noise from corridor_noisy.png and save as corridor_filtered.png.

    Steps:
      (a) split into channels
      (b) detect most noisy channel + choose median vs gaussian
      (c) merge channels
      (d) save

    Returns: filtered BGR image (uint8)
    """
    corridor_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if corridor_img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # (a) Split into channels (OpenCV uses BGR order)
    b, g, r = cv2.split(corridor_img)

    # Optional: visually inspect each channel
    if preview:
        cv2.imshow("Channel B", b)
        cv2.imshow("Channel G", g)
        cv2.imshow("Channel R", r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # (b) Identify most noisy channel using Laplacian std
    noise_scores = {"B": _estimate_noise_strength(b),
                    "G": _estimate_noise_strength(g),
                    "R": _estimate_noise_strength(r)}
    most_noisy = max(noise_scores, key=noise_scores.get)

    # Decide noise type on the most noisy channel
    ch_map = {"B": b, "G": g, "R": r}
    use_median = _looks_like_salt_and_pepper(ch_map[most_noisy])

    # Filtering choice:
    # - medianBlur is good for salt & pepper noise
    # - GaussianBlur is good for Gaussian/grainy noise
    def filt(ch: np.ndarray) -> np.ndarray:
        if use_median:
            return cv2.medianBlur(ch, 25)  # must be odd; large => stronger noise removal
        return cv2.GaussianBlur(ch, (5, 5), 0)

    # Filter only the most affected channel
    b_f, g_f, r_f = b.copy(), g.copy(), r.copy()
    if most_noisy == "B":
        b_f = filt(b)
    elif most_noisy == "G":
        g_f = filt(g)
    else:
        r_f = filt(r)

    # (c) Merge back together
    filtered = cv2.merge([b_f, g_f, r_f])

    # (d) Save result
    if not cv2.imwrite(output_path, filtered):
        raise IOError(f"Failed to write output image: {output_path}")

    if preview:
        cv2.imshow("Original", corridor_img)
        cv2.imshow("Filtered", filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered


# =============================================================================
# Cross Correlation vs Convolution (patch-based)
# =============================================================================
def correlate(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
    Cross-correlation between two same-sized 2D patches:
      sum_{r,c} patch1[r,c] * patch2[r,c]
    Implemented via explicit double-loop (slow but clear).
    """
    assert patch1.shape == patch2.shape and patch1.ndim == 2

    m, n = patch1.shape
    total = 0.0
    for r in range(m):
        for c in range(n):
            total += patch1[r, c] * patch2[r, c]
    return total


def correlate_fast(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
    Fast cross-correlation using NumPy vectorization.
    """
    assert patch1.shape == patch2.shape and patch1.ndim == 2
    return float(np.sum(patch1 * patch2))


def convolve(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
    Convolution between two same-sized 2D patches.
    Convolution = correlation with the second patch flipped both axes.
    """
    assert patch1.shape == patch2.shape and patch1.ndim == 2
    flipped = np.flipud(np.fliplr(patch2))   # flip vertically + horizontally
    return correlate(patch1, flipped)


def convolve_fast(patch1: np.ndarray, patch2: np.ndarray) -> float:
    """
    Fast convolution: same as convolve, but uses correlate_fast.
    """
    assert patch1.shape == patch2.shape and patch1.ndim == 2
    flipped = np.flipud(np.fliplr(patch2))
    return correlate_fast(patch1, flipped)


# =============================================================================
# Image filtering via convolution (slow + fast)
# =============================================================================
def filter_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D filtering to a 2D image using convolution with a given kernel.

    - Computes only the "valid" region (where kernel fully fits).
    - Leaves borders as 0 because kernel cannot be applied there.

    Returns a float64 image.
    """
    assert image.ndim == 2 and kernel.ndim == 2
    kh, kw = kernel.shape
    assert kh == kw and kh % 2 == 1, "Kernel must be odd-sized square (e.g., 3x3, 5x5)."

    out = np.zeros_like(image, dtype=np.float64)
    radius = kh // 2

    H, W = image.shape
    for r in range(radius, H - radius):
        for c in range(radius, W - radius):
            patch = image[r - radius:r + radius + 1, c - radius:c + radius + 1]
            out[r, c] = convolve(patch, kernel)
    return out


def filter_image_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Same as filter_image, but uses convolve_fast instead of convolve.
    Output should match filter_image (within floating precision).
    """
    assert image.ndim == 2 and kernel.ndim == 2
    kh, kw = kernel.shape
    assert kh % 2 == 1 and kw % 2 == 1, "Kernel must be odd-sized."

    out = np.zeros_like(image, dtype=np.float64)
    rh, rw = kh // 2, kw // 2
    H, W = image.shape

    for r in range(rh, H - rh):
        for c in range(rw, W - rw):
            patch = image[r - rh:r + rh + 1, c - rw:c + rw + 1]
            out[r, c] = convolve_fast(patch, kernel)
    return out


# =============================================================================
# Tests
# =============================================================================
def test_correlate():
    """correlate and correlate_fast must match."""
    patch1 = np.ones((10, 10)) * 127
    patch2 = np.ones((10, 10)) * 100
    print(correlate(patch1, patch2))
    print(correlate_fast(patch1, patch2))


def test_convolve():
    """convolve and convolve_fast must match."""
    patch1 = np.ones((10, 10)) * 127
    patch2 = np.ones((10, 10)) * 100
    print(convolve(patch1, patch2))
    print(convolve_fast(patch1, patch2))


def test_filter_image():
    """test for filter_image using impulse kernel."""
    np.random.seed(0)
    image = np.random.randint(0, 255, (6, 6))
    kernel = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    print(image)
    print(filter_image(image, kernel))


def test_filter_image_fast():
    """filter_image and filter_image_fast should produce same result."""
    np.random.seed(0)
    image = np.random.randint(0, 255, (100, 100)).astype(np.float64)

    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float64) / 16.0

    a = filter_image(image, kernel)
    b = filter_image_fast(image, kernel)

    print("Same result?", np.allclose(a, b))


# =============================================================================
# Compare with OpenCV filter2D on think_tank.jpeg (speed + result)
# =============================================================================
def compare_with_cv2():
    """
    Compares your filter_image_fast vs OpenCV cv2.filter2D using think_tank.jpeg.

    Notes on "result comparison":
      - Your implementation computes only valid region; borders stay 0 (uncomputed).
      - OpenCV computes all pixels using borderType padding.
      - To compare fairly, we compare ONLY the valid region (cropped).
    """
    image = img  # use real photo loaded at top

    # Example smoothing kernel (gaussian-ish blur)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float64) / 16.0

    # Run your implementation
    mine = filter_image_fast(image, kernel)

    # Run OpenCV implementation (compiled + optimized)
    cv = cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)

    # Compare only valid region
    rh, rw = kernel.shape[0] // 2, kernel.shape[1] // 2
    print("Match on valid region?",
          np.allclose(mine[rh:-rh, rw:-rw], cv[rh:-rh, rw:-rw]))

    # Timing comparison
    # (Your implementation is Python loops => much slower; OpenCV is C/C++ optimized.)
    t_mine = timeit.timeit(lambda: filter_image_fast(image, kernel), number=1)
    t_cv = timeit.timeit(lambda: cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT), number=50)

    print(f"filter_image_fast (1 run):  {t_mine:.4f} s total")
    print(f"cv2.filter2D      (50 runs): {t_cv:.4f} s total")

    # Save outputs for visual inspection (overwrites)
    cv2.imwrite("mine_filtered.png", np.clip(mine, 0, 255).astype(np.uint8))
    cv2.imwrite("cv_filtered.png",   np.clip(cv,   0, 255).astype(np.uint8))
    print("Saved mine_filtered.png and cv_filtered.png using think_tank.jpeg")


# =============================================================================
# padding to avoid zero borders (compute full image)
# =============================================================================
def filter_image_padded(image: np.ndarray, kernel: np.ndarray, pad_mode: str = "constant") -> np.ndarray:
    """
    Filters the FULL image by padding first (so borders get computed too).

    pad_mode examples:
      - "constant" (pads with 0)
      - "edge"     (repeats edge values)
      - "reflect"  (mirror padding)
    """
    assert image.ndim == 2 and kernel.ndim == 2
    kh, kw = kernel.shape
    rh, rw = kh // 2, kw // 2

    padded = np.pad(image, ((rh, rh), (rw, rw)), mode=pad_mode)
    out = np.zeros_like(image, dtype=np.float64)

    H, W = image.shape
    for r in range(H):
        for c in range(W):
            patch = padded[r:r + kh, c:c + kw]
            out[r, c] = convolve_fast(patch, kernel)
    return out


# =============================================================================
# Main runner
# =============================================================================
if __name__ == "__main__":
    # Uncomment if you have corridor_noisy.png in your folder:
    filter_corridor(preview=False)
    print("Saved corridor_filtered.png")

    # tests
    test_correlate()
    test_convolve()
    test_filter_image()
    test_filter_image_fast()

    # comparison using think_tank.jpeg
    compare_with_cv2()