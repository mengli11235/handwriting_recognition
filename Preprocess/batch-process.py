import glob
import os

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import (threshold_sauvola, threshold_otsu)

# set your working directory
os.chdir('../')

dirList = glob.glob("Labels/*fused.jpg")
for d in dirList:
    image = cv2.imread(d)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    window_size = 69
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.45)
    binary_sauvola = image > thresh_sauvola
    binary_global = image > threshold_otsu(image)

    cv_image = img_as_ubyte(binary_sauvola)
    # cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
    # cv2.imwrite(os.path.join('./Output/sauvola/', d.split('/')[-1]), cv_image)

    # ret, labels = cv2.connectedComponents(cv_image)
    # # Map component labels to hue val
    # label_hue = np.uint8(179 * labels / np.max(labels))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # # cvt to BGR for display
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # # set bg label to black
    # labeled_img[label_hue == 0] = 0
    # # cv2.imshow('labeled.png', labeled_img)
    # cv2.imwrite(os.path.join('./Output/connected/', d.split('/')[-1]), labeled_img)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv_image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    second_max_label = 1
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            second_max_label = max_label
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img2[output == second_max_label] = 255
    cv2.imwrite(os.path.join('./Output/connected2/', d.split('/')[-1]), img2)

    print("success")
