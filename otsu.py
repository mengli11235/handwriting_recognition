import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
import glob

# set your working directory
# os.chdir('D:\\darknet\\darknet-master\\build\\darknet\\x64')
os.chdir('./')


dirList = glob.glob("Input/*fused.jpg")
for d in dirList:
    # print(d)
    # read image
    img = cv2.imread(d)
    print("success")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform Otsu's method
    val = filters.threshold_otsu(img)

    hist, bins_center = exposure.histogram(img)

    plt.figure(figsize=(21.65, 16.24))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.imshow(img < val, cmap='gray', interpolation='nearest')
    plt.axis('off')
    # plt.savefig("source\\image-data\\otsu\\"+str(i)+".jpg")
    plt.savefig(d.strip('.jpg')+"_otsu.jpg")