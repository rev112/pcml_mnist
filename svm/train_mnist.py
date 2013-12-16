#!/usr/bin/env python2

# Module for training SVM with the actual MNIST data

import scipy.io
import sys
import svm as svm_module
import cross_validation
import scipy as s

def evaluate(svm, datapoints, classes):
    size = len(datapoints)
    output_classes = svm.classify_2d(datapoints)
    diff_classes = classes - output_classes
    errors = s.count_nonzero(diff_classes)
    classified_correctly = size - errors
    print "Correct: %u/%u, %.2f%%" % (classified_correctly, size,
                                      100.0 * classified_correctly/size)

DEFAULT_FILENAME = '../mnist/4svm/mp_4-9_data_prepr_shuf.mat'


def train_and_test(filename = ''):
    if filename == '':
        filename = DEFAULT_FILENAME

    d = scipy.io.loadmat(filename)

    train_datapoints = d['Xtrain']
    train_classes = d['Ytrain'].flatten()
    test_datapoints = d['Xtest']
    test_classes = d['Ytest'].flatten()

    pt_n = 6000
    print 'Dataset size:', pt_n
    #cv = cross_validation.CrossValidation(train_datapoints[:pt_n], train_classes[:pt_n], M=10)
    #cv.do_cross_validation(C=0.08, tau = 0.008)
    # (0.08, 0.008)!!!

    #cv.find_init_values()
    #cv.check_parameters()
    #sys.exit(0)

    train_datapoints = train_datapoints[:pt_n]
    train_classes = train_classes[:pt_n]

    dataset_size = len(train_datapoints)
    dim = len(train_datapoints[0])

    svm = svm_module.SVM(train_datapoints[:pt_n], train_classes[:pt_n])
    svm.set_params(C=3.2, tau=0.008)
    svm.run()
    print svm.alpha.tolist()

    print "Evaluating on a train dataset..."
    evaluate(svm, train_datapoints, train_classes)
    print "Evaluating on a test dataset..."
    evaluate(svm, test_datapoints, test_classes)

usage = '''Usage:
python train_mnist.py [<dataset_file>]

where <dataset_file> should be a .mat file with entries '[XY]Train' and '[XY]Test'

mnist/4svm/mp_4-9_data_prepr_shuf.mat file is used by default
'''

if __name__ == '__main__':
    if len(sys.argv) not in [1,2]:
        print >> sys.stderr, usage
        sys.exit(1)

    if len(sys.argv) == 1:
        sys.argv.append('')
    train_and_test(sys.argv[1])

