import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 9

image = cv2.imread('../Labels/seg_test.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray_image.png', image)
# image = cv2.imread('./gray_image.png')

# window_size = 15
# thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.45)
# binary_sauvola = image > thresh_sauvola

# plt.imshow(image, cmap=plt.cm.gray)
# cv_image = img_as_ubyte(binary_sauvola)
# cv2.imwrite('i3.png', cv_image)

hist = cv2.reduce(image, 1, cv2.REDUCE_AVG).reshape(-1)

print(len(hist))
plt.plot(hist)
plt.show()

