#!/usr/bin/env python2

# Module for training SVM with actual MNIST data

import scipy.io
import sys
import svm
import cross_validation

d = scipy.io.loadmat('../mnist/4svm/mp_4-9_data_prepr_shuf.mat')

train_datapoints = d['Xtrain']
train_classes = d['Ytrain'].flatten()
test_datapoints = d['Xtest']
test_classes = d['Ytest'].flatten()

pt_n = 1000
print 'Dataset size:', pt_n
cv = cross_validation.CrossValidation(train_datapoints[:pt_n], train_classes[:pt_n], M=10)
cv.do_cross_validation(C=2, tau = 0.1)
