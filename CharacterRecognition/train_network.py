import numpy as np
import cv2
import os
import tensorflow as tf
import csv
import datetime

from network_5 import Network

NPY_STORAGE = "./numpy/"
STATISTICS_CSV = "./checkpoints/"
now=  datetime.datetime.now()
TIME =   now.strftime("%d-%m-%Y_%H-%M-%S")
print(TIME)

CHECKPOINT_DIR = "/Users/Khmer/Developer/handwriting_recognition/CharacterRecognition/csv/" + TIME + '/'
LoadCHECKPOINT_DIR = "/Users/Khmer/Developer/handwriting_recognition/CharacterRecognition/csv/03-07-2018_11-48-28/"

try:
    os.stat(CHECKPOINT_DIR)
except:
    os.mkdir(CHECKPOINT_DIR)

try:
    os.stat(STATISTICS_CSV)
except:
    os.mkdir(STATISTICS_CSV)

trainData = np.load(NPY_STORAGE + "trainData.npy")
trainLabels = np.load(NPY_STORAGE + "trainLabels.npy")
print(trainData.shape)
validationData = np.load(NPY_STORAGE + "testData.npy")
validationLabels = np.load(NPY_STORAGE + "testLabels.npy")
print(len(validationData))
print('Data loaded')
N_BATCHES = 100  # N batches
N_ITERATIONS = 200
print(len(trainData))

MAKE_CHECKPOINT_EACH_N_ITERATIONS = 10
#CHECKPOINT_DIR = "./ckpt/network.ckpt"

PRINT_ACC_EVERY_N_EPOCHS = 5
network = Network()
#network.load_checkpoint("./ckpt/network.ckpt")

dataBatches = np.array_split(trainData, N_BATCHES)
labelBatches = np.array_split(trainLabels, N_BATCHES)

nData = len(trainData)
remainder_train = nData % N_BATCHES
batchsize = int(nData / N_BATCHES)
nBatches = int(nData / batchsize)
print('Training',nData, batchsize)

nData = len(validationData)
remainder_test = nData % N_BATCHES
batchsize = int(nData / N_BATCHES)
nBatchesValidation = int(nData / batchsize)

validationDataBatches = np.array_split(validationData, N_BATCHES)
validationLabelBatches = np.array_split(validationLabels, N_BATCHES)


# validationDataBatches = dataBatches
# validationLabelBatches = labelBatches

def save_csv(file, data):
    with open(file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=',')
        writer.writerow(float(val) for val in data)

def train():
    nBatches = len(dataBatches)
    print(nBatches)
    acc_train_saver, acc_test_saver, loss_saver= [],[],[]
    #network.loadNetwork(LoadCHECKPOINT_DIR)
    for epoch in range(N_ITERATIONS):
        acc_acum = 0
        loss_acum = 0
        for batchIdx in range(nBatches):
            if batchIdx == nBatches - 1:
                lr,acc, loss = network.train_batch(dataBatches[batchIdx][:remainder_train], labelBatches[batchIdx][:remainder_train])
            else:
                lr, acc, loss = network.train_batch(dataBatches[batchIdx], labelBatches[batchIdx])

            acc_acum = acc_acum + acc
            loss_acum = loss_acum + loss.sum()
            if batchIdx % 100 == 0:
                print( 'batch_id', batchIdx, 'acc', acc_acum/batchIdx,'loss' ,loss_acum)

        if epoch % 1 == 0:
            print("@epoch training ", epoch, "/", N_ITERATIONS, " train accuracy = ", acc_acum/nBatches," test loss = ", loss_acum)
            test_accuracy = accuracy()
            acc_train_saver.append(acc_acum/nBatches)
            acc_test_saver.append(test_accuracy)
            loss_saver.append(loss_acum)
            save_csv(STATISTICS_CSV + 'acc_train_saver'+TIME+'.csv', acc_train_saver)
            save_csv(STATISTICS_CSV + 'acc_test_saver'+TIME + '.csv', acc_test_saver)
            save_csv(STATISTICS_CSV + 'loss_saver'+TIME+'.csv', loss_saver)
            print("@epoch test ", epoch, "/", N_ITERATIONS, " _*_*_*_*_*_*_test accuracy = ", test_accuracy, ' learning rate = ', lr)
        if epoch % MAKE_CHECKPOINT_EACH_N_ITERATIONS == 0 and epoch != 0:
            network.store_checkpoint(CHECKPOINT_DIR)


def accuracy():
    # nBatches = len(validationDataBatches)

    runningAvg = 0.0

    nBatches = len(validationDataBatches)
    ##print "nData = ", nData
    # print "vallabshape ", validationLabelBatches.shape
    for batchIdx in range(nBatches):
        if batchIdx == nBatches - 1:
            acc = network.test_batch(validationDataBatches[batchIdx][:remainder_test],
                                     validationLabelBatches[batchIdx][:remainder_test])
        else:
            acc = network.test_batch(validationDataBatches[batchIdx], validationLabelBatches[batchIdx])
        runningAvg += acc[0]  # * len(validationDataBatches)
    runningAvg /= nBatchesValidation
    return runningAvg


if __name__ == "__main__":
    train()

