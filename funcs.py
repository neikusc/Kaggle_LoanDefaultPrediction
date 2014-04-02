# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import numpy as np
from sklearn.metrics import *

def load_csv(filename):    
    '''load CSV into numpy array'''
    workDir = r'/home/neik/Kaggle/LoanDefaultPrediction/'
    
    return np.genfromtxt(open(workDir + filename,'rb'), delimiter=',',skip_header=1) # 

# save data into csv format
def save_csv(filename, obj):
    workDir = r'/home/neik/Kaggle/LoanDefaultPrediction/'
    return np.savetxt(workDir + filename, obj, fmt='%d')

def load_npy(filename):
    '''load NPY into numpy array'''
    workDir = r'/home/neik/Kaggle/LoanDefaultPrediction/'
    return np.load(workDir + filename)

def save_npy(file_path, obj):
    '''Saves an object using numpy.save.'''
    workDir = r'/home/neik/Kaggle/LoanDefaultPrediction/'
    return np.save(workDir + file_path, obj)

def f1search(y_true, prob):
    '''Search for the best F1 score'''
    mx = []
    for thred in np.arange(0.55,0.75,0.001):
        y_pred = [1 if x>thred else 0 for x in prob]
        f1 = f1_score(y_true, y_pred)
        mx.append([thred, f1])
        #print thred, f1
    mx = np.array(mx)    
    thred, max_f1  = mx[np.argmax(mx[:,1]), ]
        
    return thred, max_f1

def subData(bestFeatures):
    '''load the data filled with median and extract with best features'''
    pre_train = load_npy('train_fillna.npy')
    pre_test = load_npy('test_fillna.npy')
    loss = load_npy('train_labels.npy')

    train = pre_train[:,bestFeatures]
    test = pre_test[:,bestFeatures]    
    
    return train, test, loss

def boundLoss(X):
    '''bond the loss in range 0-100 and round to integers'''
    X[np.where(X<0)[0]] = 0
    X[np.where(X>100)[0]] = 100
    X = np.round(X)
    return X

def fw_transform(X,ld):
    newX = (np.power(X,ld)-1.)/ld
    return newX

def bw_transform(X,ld):
    newX = np.power((ld*X+1.),1./ld)
    return newX
