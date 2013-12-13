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
#cv.do_cross_validation(C=2.56, tau = 0.008)
# (3.2, 0.008)!!!

#cv.find_init_values()
#cv.check_parameters()
#sys.exit(0)

train_datapoints = train_datapoints[:pt_n]
train_classes = train_classes[:pt_n]

dataset_size = len(train_datapoints)
dim = len(train_datapoints[0])

svm = svm.SVM(train_datapoints[:pt_n], train_classes[:pt_n])
svm.set_params(C=3.2, tau=0.008)
svm.run()
print svm.alpha

classified_correctly = 0
trainset_size = len(test_datapoints)
print "Evaluating on a test dataset..."
for i in range(trainset_size):
    dp = test_datapoints[i]
    true_cl = test_classes[i]
    output_class = svm.classify(dp)
    if output_class == true_cl:
       classified_correctly += 1

print "Correct: %u/%u, %.2f%%" % (classified_correctly, trainset_size,
                                 100.0 * classified_correctly/trainset_size)

