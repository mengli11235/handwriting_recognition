import csv
import random
import numpy as np
import os

NPY_STORAGE =  "/Users/Khmer/Developer/handwriting_recognition/Ngrams_LSTM/numpy/"
try:
    os.stat(NPY_STORAGE)
except:
    os.mkdir(NPY_STORAGE)
reader = csv.reader(open('./ngrams.csv', 'r'))
d = {}

# labels = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel', 'He', 'Het', 'Kaf', 'Kaf-final', 'Lamed', 'Mem', 'Mem-medial', 'Nun-final', 'Nun-medial', 'Pe', 'Pe-final', 'Qof', 'Resh', 'Samekh', 'Shin', 'Taw', 'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw', 'Yod', 'Zayin']

bigrams = []
trigrams = []
fourgrams = []
fivegrams = []

#Creat bigrams, trigrams and ngrams
for row in reader:
   aux = row[0]
   a1,a2 = aux.split(';')
   ngrams = a1.split('_')
   for i in range(0,len(ngrams)-1):
       #print(ngrams[i],i)
       for j in range(0,int(a2)):
        bigrams.append([ngrams[i],ngrams[i+1]])

   if len(ngrams) > 2:
       for i in range(0, len(ngrams) - 2):
           # print(ngrams[i],i)
           for j in range(0, int(a2)):
               trigrams.append([ngrams[i], ngrams[i + 1], ngrams[i + 2]])

   if len(ngrams) > 3:
       for i in range(0, len(ngrams) - 3):
           # print(ngrams[i],i)
           for j in range(0, int(a2)):
               fourgrams.append([ngrams[i], ngrams[i + 1], ngrams[i + 2], ngrams[i + 3]])

random.shuffle(trigrams)
random.shuffle(bigrams)
print(fourgrams)
print('Size bigrams',len(bigrams))
print('Size trigrams', len(trigrams))
print('Size trigrams', len(fourgrams))


np.save(NPY_STORAGE + "bigrams", np.asarray(bigrams))
np.save(NPY_STORAGE + "trigrams", np.asarray(trigrams))
np.save(NPY_STORAGE + "fourgrams", np.asarray(fourgrams))

