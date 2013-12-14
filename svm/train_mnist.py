#!/usr/bin/env python2

# Module for training SVM with actual MNIST data

import scipy.io
import sys
import svm
import cross_validation
import scipy as s

d = scipy.io.loadmat('../mnist/4svm/mp_4-9_data_prepr_shuf.mat')

train_datapoints = d['Xtrain']
train_classes = d['Ytrain'].flatten()
test_datapoints = d['Xtest']
test_classes = d['Ytest'].flatten()

pt_n = 6000
print 'Dataset size:', pt_n
#cv = cross_validation.CrossValidation(train_datapoints[:pt_n], train_classes[:pt_n], M=10)
#cv.do_cross_validation(C=3.2, tau = 0.008)
# (3.2, 0.008)!!!

#cv.find_init_values()
#cv.check_parameters()
#sys.exit(0)

train_datapoints = train_datapoints[:pt_n]
train_classes = train_classes[:pt_n]

dataset_size = len(train_datapoints)
dim = len(train_datapoints[0])

svm = svm.SVM(train_datapoints[:pt_n], train_classes[:pt_n])
svm.set_params(C=0.1, tau=0.008)
svm.run()
print svm.alpha.tolist()

trainset_size = len(test_datapoints)
print "Evaluating on a test dataset..."

test_output = svm.get_output_2d(test_datapoints)
print 'Test output:', test_output
classify_vect = s.vectorize(svm.classify_output)
output_classes = classify_vect(test_output)

diff_classes = test_classes - output_classes
errors = s.count_nonzero(diff_classes)
classified_correctly = trainset_size - errors

print "Correct: %u/%u, %.2f%%" % (classified_correctly, trainset_size,
                                 100.0 * classified_correctly/trainset_size)

