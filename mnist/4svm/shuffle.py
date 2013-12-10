#!/usr/bin/env python

from numpy import *
from random import shuffle
import scipy.io
import sys

def shuffle_data():
    file_name = 'mp_4-9_data_prepr.mat'
    d = scipy.io.loadmat(file_name) # corresponding MAT file
    
    # load training dataset
    training_data = d['Xtrain']
    training_classes = d['Ytrain']
    print "Data loaded."

    n = len(training_data)
    indexes = range(n)
    shuffle(indexes)
    training_data = training_data[indexes]
    training_classes = training_classes[indexes]
    
    d['Xtrain'] = training_data
    d['Ytrain'] = training_classes

    # save to new file
    file_name = file_name.replace('.mat', '_shuf.mat')
    scipy.io.savemat(file_name, d, oned_as='column')
    print "Preprocessing finished."

if __name__ == "__main__":
    shuffle_data()

