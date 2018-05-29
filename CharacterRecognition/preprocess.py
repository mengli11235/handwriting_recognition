
import cv2
import numpy as np

height = 48
width = 48
nClasses = 27
def preprocess(image):
  image = cv2.resize(image, (height, width))
  image = image.reshape((-1, height, width, 1))

  image = image.astype('float') / 255
  return image

def oneHot(size, idx):
  vec = np.zeros((size))
  vec[idx] = 1
  vec = vec.reshape((-1, nClasses))
  return vec
