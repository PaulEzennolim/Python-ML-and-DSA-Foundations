#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:40:48 2026

@author: paulezennolim

First/Second-order differential operators + Canny + Hough circles
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# First order differential operations (Sobel + magnitude)
# ============================================================
def first_order():
    """
    Produces the 2x3 figure:
      I, Ix, Iy
      ||∇I||, ||∇I||>=100, ||∇I||>=250
    """

    # 1) Read the image (OpenCV reads BGR)
    img_bgr = cv2.imread("wall.png")
    if img_bgr is None:
        raise FileNotFoundError("Could not read 'wall.png' (make sure it is in the working directory).")

    # Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) Convert to grayscale (single channel)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3) Noise reduction using Gaussian blur (helps reduce spurious edges)
    smooth_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4) Compute first derivatives using Sobel (output float to keep negatives)
    i_x = cv2.Sobel(smooth_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    i_y = cv2.Sobel(smooth_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

    # 5) Gradient magnitude ||∇I|| = sqrt(Ix^2 + Iy^2)
    magnitude = cv2.magnitude(i_x, i_y)

    # 6) Threshold the magnitude to get binary edge maps
    _, t100 = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
    _, t250 = cv2.threshold(magnitude, 250, 255, cv2.THRESH_BINARY)

    # Ensure nice display (uint8 0/255)
    t100 = t100.astype(np.uint8)
    t250 = t250.astype(np.uint8)

    # 7) Plot (2 rows x 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("I")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(i_x, cmap="gray")
    axes[0, 1].set_title("Ix")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(i_y, cmap="gray")
    axes[0, 2].set_title("Iy")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(magnitude, cmap="gray")
    axes[1, 0].set_title("||∇I||")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(t100, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("||∇I|| ≥ 100")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(t250, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("||∇I|| ≥ 250")
    axes[1, 2].axis("off")

    fig.suptitle("First order differential operations")
    plt.show()

    return img_rgb, gray, smooth_gray, i_x, i_y, magnitude, t100, t250


# ============================================================
# Helper: Zero crossings (for Laplacian sign changes)
# ============================================================
def zero_crossings(laplacian):
    """
    Returns a boolean matrix with True where the Laplacian changes sign
    (possible zero-crossings), else False.

    We check sign changes to the right and below each pixel.
    """
    zc = np.zeros_like(laplacian, dtype=bool)

    for r in range(laplacian.shape[0] - 1):
        for c in range(laplacian.shape[1] - 1):
            sig = np.sign(laplacian[r, c])

            # If sign differs to the right OR below, we mark a zero-crossing
            if np.sign(laplacian[r, c + 1]) != sig or np.sign(laplacian[r + 1, c]) != sig:
                zc[r, c] = True

    return zc


# ============================================================
# Second order differential operations (Laplacian + zero crossings)
# ============================================================
def second_order():
    """
    Produces the figure:
      Top row: I, ||∇I||>=250, ΔI
      Bottom row: ΔI≈0,  ΔI≈0 & ||∇I||>=250
    """

    # --- Read image ---
    img_bgr = cv2.imread("wall.png")
    if img_bgr is None:
        raise FileNotFoundError("Could not read 'wall.png'. Make sure it is in the working directory.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Grayscale + smoothing ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smooth_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- First order gradients + magnitude (reused) ---
    i_x = cv2.Sobel(smooth_gray, cv2.CV_64F, 1, 0, ksize=3)
    i_y = cv2.Sobel(smooth_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(i_x, i_y)

    # Threshold at 250
    _, t250 = cv2.threshold(magnitude, 250, 255, cv2.THRESH_BINARY)
    t250 = t250.astype(np.uint8)

    # --- Second order: Laplacian ΔI ---
    lap = cv2.Laplacian(smooth_gray, cv2.CV_64F, ksize=3)

    # --- Zero-crossings of Laplacian (ΔI ≈ 0 locations) ---
    zc = zero_crossings(lap)
    zc_img = (zc.astype(np.uint8) * 255)  # show as 0/255

    # Combine: (ΔI ≈ 0) AND strong gradient (||∇I|| >= 250)
    combined = (zc & (t250 > 0)).astype(np.uint8) * 255

    # --- Plot with layout matching the worksheet (bottom row centered) ---
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    ax1.imshow(img_rgb)
    ax1.set_title("I")
    ax1.axis("off")

    ax2.imshow(t250, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("||∇I|| ≥ 250")
    ax2.axis("off")

    ax3.imshow(lap, cmap="gray")
    ax3.set_title("ΔI")
    ax3.axis("off")

    ax4.imshow(zc_img, cmap="gray", vmin=0, vmax=255)
    ax4.set_title("ΔI ≈ 0")
    ax4.axis("off")

    ax5.imshow(combined, cmap="gray", vmin=0, vmax=255)
    ax5.set_title("ΔI ≈ 0 & ||∇I|| ≥ 250")
    ax5.axis("off")

    fig.suptitle("Comparison with second order differential operations")
    plt.show()

    return img_rgb, t250, lap, zc_img, combined


# ============================================================
# Comparison (thresholded magnitude vs zero-crossing+grad vs Canny)
# ============================================================
def comparison(
    gaussian_ksize=(5, 5),
    gaussian_sigma=0,
    grad_thresh=250,
    canny_low=60,
    canny_high=180,
    canny_aperture=3,
    canny_L2gradient=True
):
    """
    Produces 1x3 figure:
      1) ||∇I|| ≥ grad_thresh
      2) ΔI ≈ 0 & ||∇I|| ≥ 20
      3) Canny

    You can "play" with Canny thresholds and smoothing to improve results.
    """

    # --- Read image ---
    img_bgr = cv2.imread("wall.png")
    if img_bgr is None:
        raise FileNotFoundError("Could not read 'wall.png'. Make sure it is in the working directory.")

    # --- Gray + smoothing ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, gaussian_ksize, gaussian_sigma)

    # --- Gradient magnitude ---
    ix = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(ix, iy)

    # Panel 1: thresholded magnitude (>= grad_thresh)
    _, t_grad = cv2.threshold(mag, grad_thresh, 255, cv2.THRESH_BINARY)
    t_grad = t_grad.astype(np.uint8)

    # Panel 2: Laplacian zero-crossings AND a small gradient threshold (>= 20)
    lap = cv2.Laplacian(smooth, cv2.CV_64F, ksize=3)
    zc = zero_crossings(lap)

    _, t20 = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)
    zc_and_grad20 = (zc & (t20 > 0)).astype(np.uint8) * 255

    # Panel 3: Canny edges (expects uint8 image)
    canny_edges = cv2.Canny(
        smooth.astype(np.uint8),
        threshold1=canny_low,
        threshold2=canny_high,
        apertureSize=canny_aperture,
        L2gradient=canny_L2gradient
    )

    # --- Plot 1x3 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].imshow(t_grad, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("||∇I|| ≥ 250")
    axes[0].axis("off")

    axes[1].imshow(zc_and_grad20, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("ΔI ≈ 0 & ||∇I|| ≥ 20")
    axes[1].axis("off")

    axes[2].imshow(canny_edges, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Canny")
    axes[2].axis("off")

    fig.suptitle("Comparison with second order differential operations")
    plt.show()

    return t_grad, zc_and_grad20, canny_edges


# ============================================================
# Detect coins (Canny + HoughCircles)
# ============================================================
def detect_coins(
    img_path="coins_more.png",
    blur_ksize=(7, 7),
    blur_sigma=1.5,
    canny_low=80,
    canny_high=180,
    dp=1.2,
    minDist=35,
    param1=180,      # internal Canny high threshold used by HoughCircles
    param2=28,       # accumulator threshold (smaller -> more circles, more false positives)
    minRadius=18,
    maxRadius=40
):
    """
    Produces a 1x3 figure:
      Original | Canny | Hough circles overlay

    Tune HoughCircles parameters (dp, minDist, param1, param2, minRadius, maxRadius)
    to get the desired number of detected coins.
    """

    # --- Read image (BGR) and convert to RGB for matplotlib ---
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read '{img_path}'. Make sure it is in the working directory.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Grayscale + denoise ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, blur_ksize, blur_sigma)

    # --- Canny edges (for the middle panel) ---
    # Note: Canny expects uint8, and smooth is already uint8.
    canny_edges = cv2.Canny(smooth, canny_low, canny_high)

    # --- Hough circles (works best on smoothed grayscale) ---
    circles = cv2.HoughCircles(
        smooth,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # --- Draw circles on a copy of the original image ---
    img_hc = img_rgb.copy()

    if circles is not None:
        # HoughCircles returns (x, y, r) floats; round and convert to int
        circles_int = np.round(circles[0]).astype(int)

        for x, y, r in circles_int:
            # Draw circle outline (red) and a small center dot
            cv2.circle(img_hc, (x, y), r, (255, 0, 0), 2)
            cv2.circle(img_hc, (x, y), 2, (255, 0, 0), 2)

    # --- Plot results ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    ax[0].imshow(img_rgb)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(canny_edges, cmap="gray")
    ax[1].set_title("Canny")
    ax[1].axis("off")

    ax[2].imshow(img_hc)
    ax[2].set_title("Hough circles")
    ax[2].axis("off")

    plt.show()

    return img_rgb, canny_edges, circles, img_hc


# ============================================================
# Uncomment ONE at a time to run the function you want
# ============================================================
# first_order()
# second_order()
# comparison()
detect_coins()