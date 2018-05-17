import cv2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure

# set your working directory
os.chdir('D:\\darknet\\darknet-master\\build\\darknet\\x64')

# read image
pname = 'P21-Fg006-R-C01-R01-fused'
img = cv2.imread("source\\image-data\\"+pname+".jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform Otsu's method
val = filters.threshold_otsu(img)

hist, bins_center = exposure.histogram(img)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.imshow(img < val, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.plot(bins_center, hist, lw=2)
plt.axvline(val, color='k', ls='--')

# plt.tight_layout()
# plt.show()

# save the figure result
plt.savefig("source\\image-data\\otsu\\"+pname+".png")

# run the example dog detection using darknet, YOLOv3
# name of the function built under win10
fname = "darknet_no_gpu.exe"
p = subprocess.Popen([fname, 'detector', 'test', 'cfg\\coco.data', 'cfg\\yolov3.cfg', 'yolov3.weights', 'data\\dog.jpg'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
(output, err) = p.communicate()
print(err.decode())
if output != "":
    result = output.decode()
    print(result)