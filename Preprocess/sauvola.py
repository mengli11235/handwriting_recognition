import matplotlib
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

matplotlib.rcParams['font.size'] = 9

image = cv2.imread('../Labels/P344-Fg001-R-C01-R01-fused.jpg')
# image = cv2.imread('../Labels/P123-Fg001-R-C01-R01-fused.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray_image.png', image)
# image = cv2.imread('./gray_image.png')

binary_global = image > threshold_otsu(image)

window_size = 15
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.45)
binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola

plt.imshow(image, cmap=plt.cm.gray)
plt.show()

plt.imshow(binary_global, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_global)
cv2.imwrite('i1.png', cv_image)
plt.show()

plt.imshow(binary_niblack, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_niblack)
cv2.imwrite('i2.png', cv_image)
plt.show()

plt.imshow(binary_sauvola, cmap=plt.cm.gray)
cv_image = img_as_ubyte(binary_sauvola)
cv2.imwrite('i3.png', cv_image)
plt.show()
