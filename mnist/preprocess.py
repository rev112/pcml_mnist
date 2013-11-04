#!/usr/bin/env python

from numpy import *
import scipy.io
import sys

def preprocess():
    file_name = sys.argv[1]
    d = scipy.io.loadmat(file_name) # corresponding MAT file
    
    # load training and test datasets
    training_data = d['Xtrain'].tolist()
    test_data = d['Xtest'].tolist()
    
    print "Data loaded."
    print "Size of training set: %u, size of test set: %u" % (len(training_data),
                                                              len(test_data))
    # compute a_max, a_min
    a_max = training_data[0][0]
    a_min = a_max
    for vector in training_data:
        max_coeff = max(vector)
        a_max = max(a_max, max_coeff)
        min_coeff = min(vector)
        a_min = min(a_min, min_coeff)
    
    print "a_max = %.2f a_min = %.2f" % (a_max, a_min)
    
    # normalize
    diff = a_max - a_min
    prepr_training_data = []
    prepr_test_data = []
    prepr_data = []
    for vector in (training_data + test_data):
        prepr_vector = map(lambda x: 1.0*(x - a_min)/diff, vector)
        prepr_data.append(prepr_vector)
    
    prepr_training_data = prepr_data[:len(training_data)]
    prepr_test_data     = prepr_data[len(training_data):]
    
    d['Xtrain'] = prepr_training_data
    d['Xtest']  = prepr_test_data
    
    # save to new file
    file_name = file_name.replace('.mat', '_prepr.mat')
    scipy.io.savemat(file_name, d, oned_as='column')
    print "Preprocessing finished."

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: preprocess.py MATFILE')
        sys.exit(1)
    preprocess()

