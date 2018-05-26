import cv2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

# set your working directory
os.chdir('D:\\darknet\\darknet-master\\build\\darknet\\x64')

# run the example dog detection using darknet, YOLOv3
# name of the function built under win10
fname = "darknet_no_gpu.exe"
# p = subprocess.Popen([fname, 'detector', 'test', 'cfg\\coco.data', 'cfg\\yolov3.cfg', 'yolov3.weights', 'data\\dog.jpg'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
p = subprocess.Popen([fname, 'detector', 'train', 'data\\obj.data', 'yolo-obj.cfg', 'darknet53.conv.74', '-dont_show'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
(output, err) = p.communicate()
print(err.decode())
if output != "":
    result = output.decode()
    print(result)