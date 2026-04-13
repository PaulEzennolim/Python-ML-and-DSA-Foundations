#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:17:27 2026

@author: paulezennolim
"""

import cv2 # OpenCV
import numpy as np # Arrays (1D, 2D, and matrices)
import matplotlib.pyplot as plt # Plots

def black_image():
    # Create a 50x40 single-channel (grayscale) black image
    img = np.zeros((50, 40), dtype=np.uint8)
    
    # Save the image
    cv2.imwrite("black.png", img)

# Call the function
black_image()

def white_image():
    # Create a 50x40 single-channel white image (all values = 255)
    img = np.ones((50, 40), dtype=np.uint8) * 255
    
    # Save the image
    cv2.imwrite("white.png", img)

# Call the function
white_image()

def rgb_gray_image():
    # Create a 50x40 RGB gray image (all channels = 127)
    img = np.ones((50, 40, 3), dtype=np.uint8) * 127
    
    # Save the image
    cv2.imwrite("gray.png", img)

# Call the function
rgb_gray_image()

def yellow_image():
    # Create single-channel arrays
    r = np.ones((50, 40), dtype=np.uint8) * 255   # white (red channel)
    g = np.ones((50, 40), dtype=np.uint8) * 255   # white (green channel)
    b = np.zeros((50, 40), dtype=np.uint8)        # black (blue channel)

    # OpenCV expects B, G, R
    img = cv2.merge([b, g, r])
    
    # Save the image
    cv2.imwrite("yellow.png", img)

# Call the function
yellow_image()

def save_red():
    # Open the image
    img = cv2.imread("cam.png")
    
    # Split into B, G, R channels (OpenCV uses BGR order)
    b, g, r = cv2.split(img)
    
    # Save only the red channel
    cv2.imwrite("cam_red.png", r)

# Call the function
save_red()

def color_to_grayscale():
    # Open the image (BGR format in OpenCV)
    img = cv2.imread("cam.png")

    # Split channels
    b, g, r = cv2.split(img)

    # Apply grayscale formula (use float for accuracy)
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Convert to uint8 so OpenCV can handle it
    y = y.astype(np.uint8)

    # Display grayscale image
    cv2.imshow("Grayscale Image", y)

    # Required OpenCV window handling
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function
color_to_grayscale()

def show_with_plt_color():
    # Open the image (BGR format)
    img = cv2.imread("cam.png")

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image with matplotlib
    plt.imshow(img_rgb)
    plt.axis("off")   # optional: hides axes
    plt.show()

# Call the function
show_with_plt_color()

def show_with_plt_gray():
    # Open image directly as grayscale
    img_gray = cv2.imread("cam.png", cv2.IMREAD_GRAYSCALE)

    # Show with matplotlib using grayscale colormap
    plt.imshow(img_gray, cmap="gray")
    plt.axis("off")   # optional
    plt.show()

# Call the function
show_with_plt_gray()


def merge_two_in_one():
    # Open color image (BGR)
    img = cv2.imread("cam.png")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to 3 channels so shapes match
    gray_3ch = cv2.merge([gray, gray, gray])

    # Stack color and grayscale horizontally
    both = np.hstack((img, gray_3ch))

    # Save result
    cv2.imwrite("both.png", both)

# Call the function
merge_two_in_one()
