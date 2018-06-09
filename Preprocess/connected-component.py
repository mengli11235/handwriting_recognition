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

cv_image = img_as_ubyte(binary_sauvola)
ret, labels = cv2.connectedComponents(cv_image)

# cv_image = (255 - cv_image)
plt.imshow(cv_image, cmap=plt.cm.gray)
plt.show()

# Map component labels to hue val
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue == 0] = 0

# cv2.imshow('labeled.png', labeled_img)
plt.imshow(labeled_img, cmap=plt.cm.gray)
plt.show()
