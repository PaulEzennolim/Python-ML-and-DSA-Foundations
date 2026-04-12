#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:58:07 2026

@author: paulezennolim
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def some_stats(img_path="tree_dark_small.png"):
    # Read image in grayscale (single channel, uint8 values 0..255)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # If path is wrong / file missing, stop with a clear error
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Print simple intensity statistics for the original image
    print("Original image stats:")
    print(f"  min: {img.min():.2f}")    # darkest pixel value
    print(f"  max: {img.max():.2f}")    # brightest pixel value
    print(f"  avg: {img.mean():.2f}")   # mean brightness

    # Normalize the image to span the full range [0, 255] using min-max normalization
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)  # ensure 8-bit format

    # Print intensity statistics for the normalized image
    print("Normalized image stats:")
    print(f"  min: {img_norm.min():.2f}")
    print(f"  max: {img_norm.max():.2f}")
    print(f"  avg: {img_norm.mean():.2f}")

    # Compute histogram of original image (256 bins for pixel values 0..255)
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()

    # Compute histogram of normalized image
    img_norm_hist = cv2.calcHist([img_norm], [0], None, [256], [0, 256]).ravel()

    # Plot original image + its histogram, and normalized image + its histogram
    fig, axes = plt.subplots(2, 2)  # creates 4 plotting axes (2 rows x 2 cols)

    # Top-left: original image
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Top-right: histogram of original image
    axes[0, 1].plot(img_hist)
    axes[0, 1].set_title("Original histogram")
    axes[0, 1].set_xlim(0, 255)

    # Bottom-left: normalized image
    axes[1, 0].imshow(img_norm, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Normalized")
    axes[1, 0].axis("off")

    # Bottom-right: histogram of normalized image
    axes[1, 1].plot(img_norm_hist)
    axes[1, 1].set_title("Normalized histogram")
    axes[1, 1].set_xlim(0, 255)

    # Improve spacing so labels don’t overlap
    plt.tight_layout()
    plt.show()

# Run:
# some_stats("tree_dark_small.png")

def apply_gamma(img, gamma=1.8):
    # Gamma correction using a lookup table (LUT) for speed
    # gamma > 1 brightens dark regions; gamma < 1 darkens the image
    inv = 1.0 / gamma

    # Build LUT mapping [0..255] -> [0..255] using the gamma curve
    table = (np.linspace(0, 255, 256) / 255.0) ** inv
    table = np.clip(table * 255, 0, 255).astype(np.uint8)

    # Apply LUT to every pixel
    return cv2.LUT(img, table)

def hist_256(img):
    # Convenience function: returns a 256-bin histogram for a grayscale image
    return cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()

def unsharp_mask(img, sigma=1.2, amount=1.0):
    # Sharpening: subtract a blurred version from the original (unsharp masking)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Weighted sum: (1+amount)*img - amount*blur
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

    # Clip to valid 8-bit range
    return np.clip(sharp, 0, 255).astype(np.uint8)

def some_stats_with_enhancements(img_path="tree_dark_small.png", gamma=1.8, clahe_clip=2.0, clahe_grid=(8, 8)):
    # Load original grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Apply several contrast enhancement methods for comparison
    img_gamma = apply_gamma(img, gamma=gamma)  # gamma correction
    img_heq = cv2.equalizeHist(img)            # global histogram equalization

    # CLAHE = Contrast Limited Adaptive Histogram Equalization (local enhancement)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    img_clahe = clahe.apply(img)

    # Optional extra sharpening after CLAHE
    img_clahe_sharp = unsharp_mask(img_clahe, sigma=1.2, amount=1.0)

    # Store all versions for looping/plotting
    variants = [
        ("Original", img),
        (f"Gamma (γ={gamma})", img_gamma),
        ("Hist Equalization", img_heq),
        (f"CLAHE (clip={clahe_clip})", img_clahe),
        ("CLAHE + Unsharp", img_clahe_sharp),
    ]

    # Print stats for each variant (min/max/mean intensity)
    for name, im in variants:
        print(f"{name:18s}  min={im.min():.0f}  max={im.max():.0f}  avg={im.mean():.2f}")

    # Plot each image with its histogram (two columns: image | histogram)
    rows = len(variants)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.2 * rows))

    for r, (name, im) in enumerate(variants):
        # Left column: image
        axes[r, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_title(name)
        axes[r, 0].axis("off")

        # Right column: histogram
        axes[r, 1].plot(hist_256(im))
        axes[r, 1].set_title(f"{name} histogram")
        axes[r, 1].set_xlim(0, 255)

    plt.tight_layout()
    plt.show()

# Run:
# some_stats_with_enhancements("tree_dark_small.png", gamma=1.8)

def threshold_fingerprint(img_path="fingerprint.png", save_path="fingerprint_binary.png"):
    # Load fingerprint image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Invert so ridges become bright (helps when thresholding)
    img_inv = cv2.bitwise_not(img)

    # Binary threshold: pixels > 140 become 255, otherwise 0
    _, img_thresh = cv2.threshold(img_inv, 140, 255, cv2.THRESH_BINARY)

    # Save thresholded output to disk
    cv2.imwrite(save_path, img_thresh)

    # Display thresholded result
    plt.imshow(img_thresh, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Thresholded Fingerprint")
    plt.show()

# Run:
# threshold_fingerprint("fingerprint.png")

def advert(bg_path="diamond.jpg",
           logo_path="logo-sheffield.png",
           x=10,
           y=1080,
           save_path="advert_result.png"):

    # Load the background image (BGR color)
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(f"Could not read {bg_path}")

    # Load the logo image; IMREAD_UNCHANGED keeps alpha channel if present
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"Could not read {logo_path}")

    bg_h, bg_w = bg.shape[:2]

    # If logo has an alpha channel, separate it; otherwise treat as opaque
    if logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3]
        alpha = logo[:, :, 3] / 255.0  # convert 0..255 -> 0..1
    else:
        logo_bgr = logo
        alpha = None

    h_logo, w_logo = logo_bgr.shape[:2]

    # Adjust x/y so the logo fully fits inside the background image
    if y + h_logo > bg_h:
        y = bg_h - h_logo
    if x + w_logo > bg_w:
        x = bg_w - w_logo

    # Region-of-interest on background where the logo will go
    roi = bg[y:y+h_logo, x:x+w_logo]

    # Overlay: either hard paste (no alpha) or alpha blend (with transparency)
    if alpha is None:
        bg[y:y+h_logo, x:x+w_logo] = logo_bgr
    else:
        a = alpha[..., None]  # reshape to (h,w,1) for broadcasting
        blended = (a * logo_bgr + (1 - a) * roi).astype(np.uint8)
        bg[y:y+h_logo, x:x+w_logo] = blended

    # Save output
    cv2.imwrite(save_path, bg)

    # Display without any white border (tight)
    plt.figure(frameon=False)
    plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    plt.show()

# Run:
# advert()

def advert_penguin(bg_path="diamond.jpg",
                   penguin_path="penguin.png",
                   x=900,
                   y=20,
                   save_path="penguin_advert.png"):

    # Load background image
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError("Background not found")

    # Load penguin with alpha channel preserved
    penguin = cv2.imread(penguin_path, cv2.IMREAD_UNCHANGED)
    if penguin is None:
        raise FileNotFoundError("Penguin not found")

    # Split into BGR + alpha mask
    peng_bgr = penguin[:, :, :3]
    alpha = penguin[:, :, 3] / 255.0  # 0..1 transparency weights

    h, w = peng_bgr.shape[:2]

    # Ensure overlay fits inside background
    if y + h > bg.shape[0]:
        y = bg.shape[0] - h
    if x + w > bg.shape[1]:
        x = bg.shape[1] - w

    # Background region where the penguin will be placed
    roi = bg[y:y+h, x:x+w]

    # Alpha blend penguin over ROI
    alpha = alpha[..., None]  # (h,w) -> (h,w,1)
    blended = (alpha * peng_bgr + (1 - alpha) * roi).astype(np.uint8)
    bg[y:y+h, x:x+w] = blended

    # Save output
    cv2.imwrite(save_path, bg)

    # Display without border
    plt.figure(frameon=False)
    plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    plt.show()

# Run:
# advert_penguin()

def my_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Manual histogram equalization for 8-bit grayscale images (uint8).
    This recreates the same idea as cv2.equalizeHist without using OpenCV for HE.
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim != 2:
        raise ValueError("Expected a grayscale (2D) image")
    if img.dtype != np.uint8:
        raise ValueError("Expected dtype uint8 (8-bit image)")

    # Compute histogram counts for intensities 0..255
    hist = np.bincount(img.ravel(), minlength=256)

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Find the first non-zero CDF value (used to avoid mapping everything to >0)
    nz = np.nonzero(cdf)[0]
    if len(nz) == 0:
        return img.copy()  # degenerate case (shouldn't happen with normal images)
    cdf_min = cdf[nz[0]]

    # Total number of pixels
    n = img.size
    if n == cdf_min:
        return img.copy()  # constant image: equalization would not change it

    # Build a lookup table (LUT) using the standard HE mapping
    lut = np.round((cdf - cdf_min) * 255.0 / (n - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply the LUT to every pixel
    return lut[img]

def compare_with_opencv(img_path: str):
    # Load an image for testing/comparison
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    # Equalize using your implementation
    mine = my_histogram_equalization(img)

    # Equalize using OpenCV reference implementation
    ocv = cv2.equalizeHist(img)

    # Compare outputs
    same = np.array_equal(mine, ocv)  # exact match check
    max_diff = int(np.max(np.abs(mine.astype(np.int16) - ocv.astype(np.int16))))  # max absolute difference

    print("Same as cv2.equalizeHist?:", same)
    print("Max absolute pixel difference:", max_diff)

    # Return arrays for further inspection/plotting if needed
    return img, mine, ocv

# Example:
# img, mine, ocv = compare_with_opencv("tree_dark_small.png")
