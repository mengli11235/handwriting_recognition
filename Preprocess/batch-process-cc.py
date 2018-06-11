import glob
import os

import cv2
import numpy as np
import skimage.filters as sf
from skimage import img_as_ubyte
from skimage.filters import (threshold_sauvola)

# set your working directory
os.chdir('../')


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

    # print(threshold + immin)
    return threshold + immin


dirList = glob.glob("Labels/*fused.jpg")
for d in dirList:
    image = cv2.imread(d)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    window_size = 29
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.5)
    binary_sauvola = image > thresh_sauvola
    binary_global = image > threshold_li(image)
    # binary_global = image > sf.threshold_minimum(image)
    # binary_global = image > sf.threshold_li(image)
    # binary_global = image > threshold_otsu(image)

    cv_image = img_as_ubyte(binary_global)
    ret, labels = cv2.connectedComponents(cv_image)

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv_image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255

    cv2.imwrite('./tmp.jpg', img2)
    tmp = cv2.imread('tmp.jpg')
    im_bw = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    im_bw = 255 - im_bw
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_bw, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img3 = np.zeros(output.shape)
    img3[output == max_label] = 255

    s_img_2 = img_as_ubyte(binary_sauvola)
    s_img_2[img3 == 255] = 255
    # cv2.imwrite(os.path.join('./Output/cc_sauvola/', d.split('/')[-1]), s_img_2)
    cv2.imwrite(os.path.join('./Output/li_sauvola/', d.split('/')[-1]), s_img_2)

    print("success")
