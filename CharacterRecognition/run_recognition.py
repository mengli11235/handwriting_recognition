#!/usr/bin/env python
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import cv2
from preprocess import preprocess
from network_5 import Network
from Viterbi import viterbi
import pickle
import replace

LOAD_CHECKPOINT_DIR = "./CharacterRecognition/csv/26-06-2018_01-14-43/"

def read_data(DATADIR):
    labels = os.listdir(DATADIR)
    # print(labels)
    testData = []
    list = os.listdir(DATADIR)
    if list[0] == '.DS_Store':
        list.pop(0)  # pop .DSstore
        # print(' Deleted DSstore -+-+-++-+-+-+')
    for x in list:  # 1 iter
        im = x
        image = cv2.imread(DATADIR + '/' + im, 0)
        currentData = preprocess(image)
        testData.append(currentData)
    return (testData)


def recognition(path_data, path_output, net):
    # Load pickles
    with open('./CharacterRecognition/pickle/OneHotLabelsDict.pkl', 'rb') as f:
        OneHotLabels = pickle.load(f)

    NumberDict = {}
    for key in OneHotLabels:
        NumberDict[np.argmax(OneHotLabels[key])] = key

    with open('./CharacterRecognition/pickle/NumberDict.pkl', 'rb') as f:
        NumberDict = pickle.load(f)

    if net == -1:
        network = Network()
        network.loadNetwork(LOAD_CHECKPOINT_DIR)
    else:
        network = net

    # Read all letters
    lines = os.listdir(path_data)
    # print(lines)
    if lines[0] == '.DS_Store':
        lines.pop(0)  # pop .DSstore
        # print(' Deleted DSstore -+-+-++-+-+-+')
    lines.reverse()
    name_output = path_data.split('/')
    name_output.reverse()
    # print('name_output' + '.txt',name_output[0])
    output_txt = open(path_output + '/' + name_output[0] + '.txt', 'w')
    for line in lines:
        line_result = []
        words = os.listdir(path_data + '/' + line)
        if words[0] == '.DS_Store':
            words.pop(0)  # pop .DSstore
            # print(' Deleted DSstore -+-+-++-+-+-+')
        words.reverse()

        for word in words:
            testData = read_data(path_data + '/' + line + '/' + word)
            testData.reverse()
            word_result = []

            for i in range(len(testData)):
                word_result.append(NumberDict[np.argmax(network.feed_batch(testData[i]))])

            # word_result = viterbi((word_result))
            line_result.append(word_result)

        for line_write in line_result:
            for word_write in line_write:
                if word_write == line_write[-1]:
                    output_txt.write(word_write)
                else:
                    output_txt.write(word_write + '-')
            output_txt.write(" ")
        output_txt.write("\n")
        # print(line_result, line)
    print('Recognition done for image' + name_output[0])
    output_txt.close()
    return network


if __name__ == '__main__':
    DIR_OUTPUT = "./"
    network = -1

    for folder in [name for name in os.listdir("./segmentation/") if os.path.isdir('./segmentation/' + name)]:
        DIR_DATA = "./segmentation/" + folder
        network = recognition(DIR_DATA, DIR_OUTPUT, network)
        replace.trans(folder)


