import cv2
import matplotlib.pyplot as plt
import numpy as np
import timeit
## read camera man's image
from skimage import data, img_as_float
import seaborn as sns
from skimage import exposure

## some helper functions defined here
def plot_img_and_hist(image, axes, bins=256,title =''):
    """Plot an image along with its histogram"""
    image = img_as_float(image)
    ax_img, ax_hist = axes

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_title(title)


    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])
    return ax_img, ax_hist

"""Below is a function named {correlate(patch1, patch2):} that implements the cross-correlation between 2 image patches, i.e.:
$$
\text{result} = \sum_{r=0}^{m-1}\sum_{c=0}^{n-1} I(r, c)\times J(r, c)
$$
In summary, it:  

- Receives 2 $m \times n$ matrices (i.e.\ 2D arrays); Here, we assume two patches are of the same size.
- Iterates over the rows and columns (double loop), multiplies the values of the matrices in corresponding positions and accumulates the results of all the multiplications to get the value.
"""

# Implementing a function to compute the cross correlation between two patches.
def correlate(patch1, patch2):
    #correlates two patches of same shape and 1 channel
    height, width = patch1.shape
    #result is sum, will add to this later
    output = 0

    #iterate through rows(height) and columns (width)
    #############WRITE YOUR CODE HERE##############



    return output #return total


#This next version is given by us.
#Uses numPy with easy syntax.
def correlate_fast(patch1, patch2):
    # checks if both patches have the same shape and are 2-dimensional
    assert patch1.shape == patch2.shape and patch1.ndim == 2
    return np.sum(patch1 * patch2)

#Test with this, should return same results.
def test_correlate():
    patch1 = np.ones((10, 10)) * 127
    patch2 = np.ones((10, 10)) * 100
    a = correlate(patch1, patch2)
    b = correlate_fast(patch1, patch2)
    assert a == b, 'test failed'
    print("test passed!")

test_correlate()

"""Now let's work on convolution.
Please implement  a function named `convolve(patch1, patch2)` that performs a convolution operation. Here, we are implementing in a smart way, i.e., using the function we just implemented above \texttt{correlate}. The function should be implemented in the following steps:

   - Take two matrices (i.e., 2D arrays) of the same dimensions as input.
   - Flip the second matrix both vertically and horizontally (think about why?).
   -  Call the function `correlate` with the appropriate arguments to compute the convolution.
   -  Return the resulting total.
"""

#As opposed to cross-correlation, kernel is flipped
#before computing element-wise multiplication
#used for image manipulation rather than matching.

def convolve(patch1, patch2):
    #############WRITE YOUR CODE HERE##############



def convolve_fast(patch1, patch2):
    #############WRITE YOUR CODE HERE##############


#Test our new functions
def test_convolve():
    patch1 = np.ones((10, 10)) * 127
    patch2 = np.ones((10, 10)) * 100
    a = convolve(patch1, patch2)
    b=convolve_fast(patch1, patch2)
    assert a==b, 'test failed'
    print("test passed!")

test_convolve()

"""######################################################################
#GROUP III Image Filtering with Customized Filters
######################################################################

Congratulations! You have implemented the convolution operation! Next, let's try how can we leverage it to implement image filtering with more customized kernels yourself !
Implement a function named \texttt{filter\_image} that applies a convolution operation to an image using a given kernel. The function should follow these steps:
1. Take two inputs: an image and a kernel (both represented as 2D NumPy arrays).
2. Create a new image with the same shape as the input image, initialized with zeros.
3. Compute the radius of the kernel based on its size.
4. Iterate over each valid pixel position in the image where the kernel can be applied. For each pixel:
  -  Extract a patch from the image, centered at the current pixel, with the same dimensions as the kernel.
  - Apply convolution by computing the element-wise product between the patch and the kernel, then summing the result. (\textbf{Hint}: you can use the $convolve$ function that you just implemented)     
  - Store the computed value in the corresponding pixel of the new image.
  - Return the new image as filtered result.
"""

#Convolves kernel around pixel
#Receives image and kernel
def filter_image(image, kernel):
    #############WRITE YOUR CODE HERE##############











#test our function on different kernels using given function
def test_filter_image():
    np.random.seed(0) # always get the same random numbers
    image = np.random.randint(0, 255, (7, 7))
    # (a)
    kernel = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]) # impulse kernel

    # (b)
    # kernel = np.array([[0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0],
    #                     [0, 0, 1, 0, 0],
    #                     [0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0]]) # impulse kernel

    # (c)
    # kernel = np.ones((3, 3)) / 9

    # (d)
    # kernel = np.array([[0, 0, 0, 0, 0],
    #                     [0, 0, 1, 0, 0],
    #                     [0, 0, 0, 0, 0]])

    print(image)
    print(filter_image(image, kernel))

test_filter_image()

"""######################################################################
#GROUP IV : Test Image Filtering with Fast Convolution
######################################################################
"""



"""Implement a function named `filter_image_fast` that is exactly the same as filter_image,
however, it calls `convolve_fast` instead of `convolve`. The result should be exactly the same
no matter which is used (i.e. filter_image_fast or filter_image).
"""

#Filter image fast
#Same as other code but just with "fast" convolution function
def filter_image_fast(image, kernel):
    #############WRITE YOUR CODE HERE##############

#Compare filter iterative vs fast using given function
#Times compilation
#Need to import timeit
def compare_convs():
    kernel = np.ones((3, 3))/9 # 3×3 average kernel
    image = data.camera()

    start_time = timeit.default_timer()
    out = cv2.filter2D(image, -1, kernel) # -1 means: out same type as image
    elapsed = timeit.default_timer() - start_time
    print("OpenCV's filter:", elapsed)

    start_time = timeit.default_timer()
    out = filter_image_fast(image, kernel)
    elapsed = timeit.default_timer() - start_time
    print("Fast filter:", elapsed)

    start_time = timeit.default_timer()
    out = filter_image(image, kernel)
    elapsed = timeit.default_timer() - start_time
    print("Slow filter:", elapsed)

compare_convs()

"""Now let's work on some real images with different types of noises and apply your and openCV's filters and compare the result."""

## add noises
def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        noisy_image (numpy.ndarray): Image with added Gaussian noise.
    """
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.float32)

    # Add noise to the image
    noisy_image = image.astype(np.float32) + gaussian_noise

    # Clip values to maintain valid range [0, 255] and convert to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds salt-and-pepper noise to an image.

    Parameters:
        image (numpy.ndarray): Input image.
        salt_prob (float): Probability of salt noise (white pixels).
        pepper_prob (float): Probability of pepper noise (black pixels).

    Returns:
        noisy_image (numpy.ndarray): Image with added salt-and-pepper noise.
    """
    noisy_image = image.copy()
    total_pixels = image.size

    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0

    return noisy_image

image = data.camera()

g_noised = add_gaussian_noise(image, mean=0, std=25)
sp_noised = add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)
fig,ax = plt.subplots(3,2,figsize=(10,10))

plot_img_and_hist(image,axes=ax[0],title = 'clean')
plot_img_and_hist(g_noised,axes=ax[1],title = 'w/ Gaussian noise')
plot_img_and_hist(sp_noised,axes=ax[2],title = 'w/ S&P noise')

kernel = np.ones((3, 3))/9 # 3×3 average kernel
image = data.camera()
cv_out = cv2.filter2D(g_noised, -1, kernel) # -1 means: out same type as image
sp_out = cv2.filter2D(sp_noised, -1, kernel) # -1 means: out same type as image

my_out = filter_image_fast(g_noised, kernel)
my_sp_out = filter_image_fast(sp_noised, kernel)

fig,ax = plt.subplots(6,2,figsize=(20,10))

plot_img_and_hist(g_noised,axes=ax[0],title = 'w/ Gaussian noise')
plot_img_and_hist(cv_out,axes=ax[1],title = 'apply CV filter to filter Gaussian noises')
plot_img_and_hist(my_out,axes=ax[2],title = 'apply My filter to filter Gaussian noises')
plot_img_and_hist(sp_noised,axes=ax[3],title = 'w/ S&P noise')
plot_img_and_hist(sp_out,axes=ax[4],title = 'apply CV filter to filter S&P noises')
plot_img_and_hist(my_sp_out,axes=ax[5],title = 'apply My filter to filter S&P noises')

"""Challenge: You may notice your filtered images all contain zeros across the borders as the kernel cannot be fully applied. To handle this, one could pad the image beforehand (e.g., using $numpy.pad$). This task is optional.  Try and Post your results in the discussion board!"""

