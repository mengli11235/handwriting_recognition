import glob
import os

import cv2
from skimage import img_as_ubyte
from skimage.filters import (threshold_sauvola)

# set your working directory
os.chdir('../')

dirList = glob.glob("Labels/*fused.jpg")
for d in dirList:
    image = cv2.imread(d)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    window_size = 25
    thresh_sauvola = threshold_sauvola(image, window_size=window_size, k=0.45)
    binary_sauvola = image > thresh_sauvola

    cv_image = img_as_ubyte(binary_sauvola)
    cv2.imwrite(os.path.join('./Output/sauvola/', d.split('/')[-1]), cv_image)
    print("success")
