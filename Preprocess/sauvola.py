import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

matplotlib.rcParams['font.size'] = 9

image = cv2.imread('../Input/P166-Fg002-R-C01-R01-fused.jpg')
# image = cv2.imread('../Input/P123-Fg001-R-C01-R01-fused.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# cv2.imwrite('gray_image.png', image)
# image = cv2.imread('./gray_image.png')


def threshold_li(image):
    """Return threshold value based on adaptation of Li's Minimum Cross Entropy method.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels with an intensity higher than
        this value are assumed to be foreground.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           DOI:10.1016/0031-3203(93)90115-D
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8): 771-776
           DOI:10.1016/S0167-8655(98)00057-9
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           DOI:10.1117/1.1631315
    .. [4] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    """
    # Make sure image has more than one value
    if np.all(image == image.flat[0]):
        raise ValueError("threshold_li is expected to work with images "
                         "having more than one value. The input image seems "
                         "to have just one value {0}.".format(image.flat[0]))

    # Copy to ensure input image is not modified
    image = image.copy()
    # Requires positive image (because of log(mean))
    immin = np.min(image)
    image -= immin
    imrange = np.max(image)
    tolerance = 20 * imrange / 256

    # Calculate the mean gray-level
    mean = np.mean(image)

    # Initial estimate
    new_thresh = mean
    old_thresh = new_thresh + 2 * tolerance

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(new_thresh - old_thresh) > tolerance:
        old_thresh = new_thresh
        threshold = old_thresh + tolerance  # range
        # Calculate the means of background and object pixels
        mean_back = image[image <= threshold].mean()
        # print(mean_back)
        mean_obj = image[image > threshold].mean()
        # print(mean_obj)

        temp = (mean_back - mean_obj) / (np.log(mean_back) - np.log(mean_obj))

        if temp < 0:
            new_thresh = temp - tolerance
        else:
            new_thresh = temp + tolerance

    print(threshold + immin)
    return threshold + immin


# binary_global = image > threshold_otsu(image)
binary_global = image > threshold_li(image)

window_size = 15
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.5)
thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.45)
binary_niblack = image > threshold_otsu(image)
binary_sauvola = image > thresh_sauvola

new_img = binary_sauvola + binary_global
plt.imshow(new_img, cmap=plt.cm.gray)
plt.show()

plt.imshow(image, cmap=plt.cm.gray)
plt.show()

plt.imshow(binary_global, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_global)
# cv2.imwrite('i1.png', cv_image)
plt.show()

plt.imshow(binary_niblack, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_niblack)
# cv2.imwrite('i2.png', cv_image)
plt.show()

plt.imshow(binary_sauvola, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_sauvola)
# cv2.imwrite('i3.png', cv_image)
plt.show()
