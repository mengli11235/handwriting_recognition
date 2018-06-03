
# coding: utf-8

# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# # Confusion matrix
# 


# In[24]:


#print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

y_true = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 6,17,18,18,19,20,21,21,23,24,25,26]
y_pred = [4,1,2,3,4,5,6,7,8,9,10, 0,12,13,14,15,16,17,18,18,19,20,21,21,23,24,25,26]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if normalize:
     #   cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
   # else:
 
    
    print('Confusion matrix, without normalization')

    print(cm)
     
    classes = [ 'Alef',  
                'Ayin', 
                'Bet',
                'Dalet',
                'Gimel',
                'He',
                'Het',
                'Kaf',
                'Kaf-final',
                'Lamed',
                'Mem',
                'Mem-medial',
                'Nun-final',
                'Nun-medial',
                'Pe',
                'Pe-final',
                'Qof',
                'Resh',
                'Samekh',
                'Shin',
                'Taw',
                'Tet',
                'Tsadi-final',
                'Tsadi-medial',
                'Waw',
                'Yod',
                'Zayin'
    ]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90)
    plt.yticks(tick_marks, classes )

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
 #                     title='Normalized confusion matrix')

plt.show()

