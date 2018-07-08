import glob
import math
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import *

from Preprocess.tools.peakdetect import *

# dirList = glob.glob("../Labels/*fused.jpg")
dirList = glob.glob("../Labels/P168-Fg016-R-C01-R01-fused.jpg")


# dirList = glob.glob("../Labels/P123-Fg002-R-C01-R01-fused.jpg")
# dirList = glob.glob('/Users/Khmer/Downloads/sample-test/run_test/*.pgm')


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
        # cv2.imwrite('./tr_' + str(d) + '.jpg', rotated_image)
        ri_hist = cv2.reduce(rotated_image, 1, cv2.REDUCE_AVG).reshape(-1)

        # plt.plot(ri_hist)
        # plt.savefig('./tr_' + str(d) + '_h.jpg')
        # plt.clf()
        # plt.show()

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
        # print("score: ", score, " degree: ", d)
        # print(": ", score_ne / num_ne, " : ", score_po / num_po)

        if score < min_score:
            degree = d
            min_score = score

    # print('Degree: ', degree)
    rotated_image = rotate_max_area(image, degree)
    # plt.imshow(rotated_image, cmap=plt.cm.gray)
    # plt.show()
    return rotated_image


def separate_cha_2(line):
    line_hist = cv2.reduce(line, 0, cv2.REDUCE_AVG).reshape(-1)
    new_line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    line_peaks = peakdetect(line_hist, lookahead=20)
    Hl, Wl = new_line.shape[:2]

    cha = []
    # for y in line_peaks[0]:
    # plt.plot(y[0], y[1], "r*")
    # cv2.line(new_line, (y[0], 0), (y[0], Hl), (255, 0, 0), 3)

    for y in line_peaks[1]:
        cha.append(y[0])
        # plt.plot(y[0], y[1], "g*")
        cv2.line(new_line, (y[0], 0), (y[0], Hl), (0, 255, 0), 3)

    cha.insert(0, 0)
    cha.append(Wl)

    plt.imshow(new_line, cmap=plt.cm.gray)
    plt.show()
    return cha


def separate_cha(line):
    line_hist = cv2.reduce(line, 0, cv2.REDUCE_AVG).reshape(-1)
    new_line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    line_peaks = peakdetect(line_hist, lookahead=25)
    Hl, Wl = new_line.shape[:2]

    cha = []
    # for y in line_peaks[0]:
    # plt.plot(y[0], y[1], "r*")
    # cv2.line(new_line, (y[0], 0), (y[0], Hl), (255, 0, 0), 3)

    for y in line_peaks[0]:
        if y[1] >= 235:
            cha.append(y[0])
        # plt.plot(y[0], y[1], "g*")
        cv2.line(new_line, (y[0], 0), (y[0], Hl), (0, 255, 0), 3)

    cha.insert(0, 0)
    cha.append(Wl)

    # plt.plot(line_hist)
    # plt.show()
    # plt.imshow(new_line, cmap=plt.cm.gray)
    # plt.show()
    return cha


def separate_words(line):
    line_hist = cv2.reduce(line, 0, cv2.REDUCE_AVG).reshape(-1)
    new_line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    line_peaks = peakdetect(line_hist, lookahead=40)
    Hl, Wl = new_line.shape[:2]

    words = []
    for y in line_peaks[0]:
        if y[1] == 255:
            words.append(y[0])
        # plt.plot(y[0], y[1], "r*")
        if y[1] == 255:
            cv2.line(new_line, (y[0], 0), (y[0], Hl), (255, 0, 0), 3)
    # for y in line_peaks[1]:
    # plt.plot(y[0], y[1], "g*")
    # if y[1] == 255:
    #     words.append(y[0])
    #     cv2.line(new_line, (y[0], 0), (y[0], Hl), (0, 255, 0), 3)

    # words.insert(0, 0)
    # words.append(Wl)

    plt.imshow(new_line, cmap=plt.cm.gray)
    plt.show()
    return words


def crop_blank(img):
    min_x, max_x, min_y, max_y = 0, 0, 0, 0

    # for line in img:
    #     wl = True
    #     for x in line:
    #         if x != 255:
    #             wl = False

    th, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    (cx, cy), (w, h), ang = ret
    if w < h:
        crop = img[int(w):int(h), :]
    else:
        crop = img[int(h):int(w), :]

    plt.imshow(crop, cmap=plt.cm.gray)
    plt.show()

    # if x < y:
    #     if w < h:
    #         crop = img[w:h, x:y]
    #     else:
    #         crop = img[h:w, x:y]
    # else:
    #     if w < h:
    #         crop = img[w:h, y:x]
    #     else:
    #         crop = img[h:w, y:x]
    #


for d in dirList:
    image = cv2.imread(d)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    window_size = 59
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.5)
    binary_sauvola = image > thresh_sauvola
    # binary_global = image > threshold_triangle(image)
    # binary_global = image > threshold_li(image)
    # binary_global = image > threshold_minimum(image)
    binary_global = image > threshold_li(image)
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

    # cv2.imwrite('./t1.jpg', cv_image)

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

    # cv2.imwrite('./t2.jpg', img2)

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

    # cv2.imwrite('./t3.jpg', img3)

    s_img_2 = img_as_ubyte(binary_sauvola)

    # cv2.imwrite('./t1_2.jpg', s_img_2)

    s_img_2[img3 == 255] = 255

    # cv2.imwrite('./t4.jpg', s_img_2)

    new_img = cv2.cvtColor(s_img_2, cv2.COLOR_GRAY2BGR)
    rotated = find_degree(new_img)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)
    H, W = rotated.shape[:2]

    peaks = peakdetect(hist, lookahead=40)
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

    peak = []
    for y in peaks[0]:
        peak.append(y[0])
        # plt.plot(y[0], y[1], "r*")
        cv2.line(rotated, (0, y[0]), (W, y[0]), (255, 0, 0), 3)
    # for y in peaks[1]:
    # peak.append(y[0])
    # plt.plot(y[0], y[1], "g*")
    # cv2.line(rotated, (0, y[0]), (W, y[0]), (0, 255, 0), 3)

    # plt.plot(hist)
    # plt.savefig('hist.jpg')
    # plt.clf()

    peak.insert(0, 0)
    peak.append(W)

    # print(peak)

    # plt.plot(hist)
    # plt.show()

    if not os.path.exists(os.path.splitext(d.split('/')[-1])[0]):
        os.makedirs(os.path.splitext(d.split('/')[-1])[0])
    else:
        shutil.rmtree(os.path.splitext(d.split('/')[-1])[0])
        os.makedirs(os.path.splitext(d.split('/')[-1])[0])

    # cv2.imwrite(os.path.join(os.path.splitext(d.split('/')[-1])[0], '_t.jpg'), rotated)
    # crop_blank(rotated)

    count_line = 0
    for y in range(len(peak) - 1):
        if not os.path.exists(os.path.join(os.path.splitext(d.split('/')[-1])[0], 'line_' + str(count_line))):
            os.makedirs(os.path.join(os.path.splitext(d.split('/')[-1])[0], 'line_' + str(count_line)))
        else:
            shutil.rmtree(os.path.join(os.path.splitext(d.split('/')[-1])[0], 'line_' + str(count_line)))
            os.makedirs(os.path.join(os.path.splitext(d.split('/')[-1])[0], 'line_' + str(count_line)))
        path = os.path.join(os.path.splitext(d.split('/')[-1])[0], 'line_' + str(count_line))

        crop_img = s_img_2[peak[y]:peak[y + 1], 0:W]

        word_peaks = separate_words(crop_img)
        if len(word_peaks) == 0:
            continue

        count_line += 1
        # plt.imshow(crop_img, cmap=plt.cm.gray)
        # plt.show()

        for i in range(len(word_peaks) - 1):
            new_w = crop_img[:, word_peaks[i]: word_peaks[i + 1]]
            os.makedirs(os.path.join(path, 'word_' + str(i)))

            cv2.line(rotated, (word_peaks[i], peak[y]), (word_peaks[i], peak[y + 1]), (0, 0, 255), 3)
            # print(y0, y[0], word_peaks[i])

            cha_peaks = separate_cha(new_w)
            if len(cha_peaks) == 0:
                continue

            for j in range(len(cha_peaks) - 1):
                new_c = new_w[:, cha_peaks[j]: cha_peaks[j + 1]]
                cv2.imwrite(os.path.join(os.path.join(path, 'word_' + str(i)), str(j) + '.jpg'),
                            new_c)

    plt.imshow(rotated, cmap=plt.cm.gray)
    plt.show()
    print("Successfully process image " + d.split('/')[-1].split('jpg')[0])
