import math

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola

matplotlib.rcParams['font.size'] = 9


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


image = cv2.imread('../Input/P123-Fg002-R-C01-R01-fused.jpg')
# image = cv2.imread('../Input/P583-Fg006-R-C01-R01-fused.jpg')
# image = cv2.imread('../Input/P25-Fg001.pgm')
kernel = np.ones((5, 5), np.float32) / 25
image = cv2.filter2D(image, -1, kernel)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

window_size = 51
thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.5)
binary_sauvola = image > thresh_sauvola
# binary_global = image > threshold_li(image)
# binary_global = image > sf.threshold_minimum(image)
# binary_global = image > sf.threshold_li(image)
# binary_global = image > skimage.filters.threshold_otsu(image)
binary_global = image > skimage.filters.threshold_mean(image)

cv_image = img_as_ubyte(binary_global)
ret, labels = cv2.connectedComponents(cv_image)

plt.imshow(binary_global, cmap=plt.cm.gray)
plt.show()

# image = 255 - image
# plt.imshow(image, cmap=plt.cm.gray)
# plt.show()
# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
# sizes = stats[:, -1]
# max_label = 1
# max_size = sizes[1]
# for i in range(2, nb_components):
#     if sizes[i] > max_size:
#         max_label = i
#         max_size = sizes[i]
# img2 = np.zeros(output.shape)
# img2[output == max_label] = 255
# cv2.imwrite('./tmp.jpg', img2)
# tmp = cv2.imread('tmp.jpg')
# im_bw = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
# plt.imshow(im_bw, cmap=plt.cm.gray)
# plt.show()

# Map component labels to hue val
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue == 0] = 0

plt.imshow(cv_image, cmap=plt.cm.gray)
plt.show()

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

plt.imshow(im_bw, cmap=plt.cm.gray)
plt.show()

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
cv2.imwrite('./tmp.jpg', s_img_2)

plt.imshow(s_img_2, cmap=plt.cm.gray)
plt.show()

hist = cv2.reduce(s_img_2, 1, cv2.REDUCE_AVG).reshape(-1)

th = 250
H, W = s_img_2.shape[:2]
uppers = [y for y in range(H - 1) if hist[y] <= th < hist[y + 1]]
lowers = [y for y in range(H - 1) if hist[y] > th >= hist[y + 1]]

rotated = cv2.cvtColor(s_img_2, cv2.COLOR_GRAY2BGR)
# for y in uppers:
#     cv2.line(rotated, (0, y), (W, y), (255, 0, 0), 1)
#
# for y in lowers:
#     cv2.line(rotated, (0, y), (W, y), (0, 255, 0), 1)

from Preprocess.tools.peakdetect import *

peaks = peakdetect(hist, lookahead=30)

for y in peaks[0]:
    plt.plot(y[0], y[1], "r*")
    cv2.line(rotated, (0, y[0]), (W, y[0]), (255, 0, 0), 3)
for y in peaks[1]:
    plt.plot(y[0], y[1], "g*")
    cv2.line(rotated, (0, y[0]), (W, y[0]), (0, 255, 0), 3)
    # print(y)
plt.plot(hist)
plt.show()

# indexes = ss.find_peaks_cwt(hist, np.arange(1, 300))
# for y in indexes:
#     cv2.line(rotated, (0, y), (W, y), (0, 255, 0), 1)
# print(indexes)

# cv2.imwrite("tmp.jpg", rotated)
plt.imshow(rotated, cmap=plt.cm.gray)
plt.show()


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_bound(image, angle):
    # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def rotate_max_area(image, angle):
    """ image: cv2 image matrix object
        angle: in degree
    """
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
    rotated = rotate_bound(image, angle)
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]


def find_degree(image):
    min_score = 999999
    degree = 0

    for d in range(-6, 7):
        rotated_image = rotate_max_area(image, d)
        ri_hist = cv2.reduce(rotated_image, 1, cv2.REDUCE_AVG).reshape(-1)
        line_peaks = peakdetect(ri_hist, lookahead=30)
        score_ne = num_ne = 0
        score_po = num_po = 0

        for y in line_peaks[0]:
            score_ne -= (y[1] * 1)
            num_ne += 1
        for y in line_peaks[1]:
            score_po += (y[1] * 1)
            num_po += 1

        score = score_ne / num_ne + score_po / num_po
        print("score: ", score, " degree: ", d)
        print(": ", score_ne / num_ne, " : ", score_po / num_po)

        if score < min_score:
            degree = d
            min_score = score

    print(degree)
    rotated_image = rotate_max_area(image, degree)
    plt.imshow(rotated_image, cmap=plt.cm.gray)
    plt.show()


s2 = cv2.cvtColor(s_img_2, cv2.COLOR_GRAY2BGR)
rotate_max_area(s2, 5)
find_degree(s2)


def separate_words(line):
    line_hist = cv2.reduce(line, 0, cv2.REDUCE_AVG).reshape(-1)
    new_line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    line_peaks = peakdetect(line_hist, lookahead=20)
    Hl, Wl = new_line.shape[:2]

    for y in line_peaks[0]:
        plt.plot(y[0], y[1], "r*")
        cv2.line(new_line, (y[0], 0), (y[0], Hl), (255, 0, 0), 3)
    for y in line_peaks[1]:
        plt.plot(y[0], y[1], "g*")
        # cv2.line(new_line, (y[0], 0), (y[0], Hl), (0, 255, 0), 3)
        # print(y)
    plt.plot(line_hist)
    plt.show()
    return new_line

#
# y0 = 0
# for y in peaks[0]:
#     crop_img = s_img_2[y0:y[0], 0:W]
#     y0 = y[0]
#     crop_img = separate_words(crop_img)
#     plt.imshow(crop_img, cmap=plt.cm.gray)
#     plt.show()
#
#     if y[0] == peaks[0][-1][0]:
#         crop_img = s_img_2[y[0]:H, 0:W]
#         crop_img = separate_words(crop_img)
#         plt.imshow(crop_img, cmap=plt.cm.gray)
#         plt.show()
#         # print('last')
