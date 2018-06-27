import numpy as np
import cv2
import os
from preprocess import preprocess
from network_5 import Network
from Viterbi import viterbi
import pickle

LOAD_CHECKPOINT_DIR = "./csv/26-06-2018_01-14-43/"


def read_data(DATADIR):
    labels = os.listdir(DATADIR)
    print(labels)
    testData = []
    list = os.listdir(DATADIR)
    if list[0] == '.DS_Store':
        list.pop(0)  # pop .DSstore
        print(' Deleted DSstore -+-+-++-+-+-+')
    for x in list:  # 1 iter
        im = x
        image = cv2.imread(DATADIR + '/' + im, 0)
        currentData = preprocess(image)
        testData.append(currentData)
    return(testData)

def recognition(path):

    #Load pickles
    with open('././pickle/OneHotLabelsDict.pkl', 'rb') as f:
        OneHotLabels = pickle.load(f)

    NumberDict = {}
    for key in OneHotLabels:
        NumberDict[np.argmax(OneHotLabels[key])] = key

    with open('./pickle/NumberDict.pkl', 'rb') as f:
        NumberDict= pickle.load(f)

    testData = read_data(DIR_DATA)
    network = Network()
    network.loadNetwork(LOAD_CHECKPOINT_DIR)
    result = []

    for i in range(len(testData)):
        result.append(NumberDict[np.argmax(network.feed_batch(testData[i]))])

    print('result',result)

    print(viterbi((result)))

    file = open('./testfile.txt', 'w')
    for i in range(len(result)):
        file.write(result[i] + ' ')

if __name__ == '__main__':
    DIR_DATA = "./monkbrill_171005_jpg_test/Alef"
    recognition(DIR_DATA)
