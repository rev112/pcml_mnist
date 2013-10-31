from numpy import *
import scipy.io
import sys

d = scipy.io.loadmat(sys.argv[1]) # corresponding MAT file
data = d['Xtrain']    # Xtest for test data
labels = d['Ytrain']  # Ytest for test labels

print 'Finished loading',data.shape[0],'datapoints'
