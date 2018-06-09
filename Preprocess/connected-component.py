import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola, threshold_otsu

matplotlib.rcParams['font.size'] = 9

# image = cv2.imread('../Labels/seg_test.jpg')
image = cv2.imread('../Labels/P344-Fg001-R-C01-R01-fused.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray_image.png', image)
# image = cv2.imread('./gray_image.png')

window_size = 69
thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.9)
binary_sauvola = image > thresh_sauvola
binary_global = image > threshold_otsu(image)

cv_image = img_as_ubyte(binary_global)
ret, labels = cv2.connectedComponents(cv_image)

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv_image, connectivity=8)
sizes = stats[:, -1]
max_label = 1
max_size = sizes[1]
for i in range(2, nb_components):
    if sizes[i] > max_size:
        max_label = i
        max_size = sizes[i]
img2 = np.zeros(output.shape)
img2[output == max_label] = 255

# cv_image = (255 - cv_image)
# plt.imshow(cv_image, cmap=plt.cm.gray)
# plt.show()

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

# cv2.imshow('labeled.png', labeled_img)
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()
