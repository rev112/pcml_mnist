#!/usr/bin/env python2

import scipy as s
import svm

class CrossValidation:

    def __init__(self, X, T, M=10):
        self.X = X
        self.T = T
        self.n = len(X)
        self.d = len(X[0])
        assert self.n == len(T)
        self.M = M

    def split_by_index(self, i):
        """Return a tuple of two dictionaries with
        keys 'dtp' (datapoints) and 'cl' (classes):
            1. training data (without part i)
            2. validation data (part i)"""
        assert 1 <= i and i <= self.M, "Invalid part number"
        T = self.T
        X = self.X
        part_size = self.n / self.M
        lbound = part_size * (i-1)
        hbound = part_size * i
        dtp_training = X[:lbound] + X[hbound:]
        cl_training = T[:lbound] + T[hbound:]
        dtp_validation = X[lbound:hbound]
        cl_validation = T[lbound:hbound]
        training_set = {'dtp': dtp_training, 'cl': cl_training}
        validation_set = {'dtp': dtp_validation, 'cl': cl_validation}
        return (training_set, validation_set)

    def do_cross_validation(self):
        svm_list = []
        for i in xrange(self.M):
            tr_set_i, val_set_i = self.split_by_index(i+1)
            tr_set_size = len(tr_set_i['dtp'])
            svm_i = svm.SVM(tr_set_size, self.d, tr_set_i['dtp'], tr_set_i['cl'])
            svm_list.append(svm_i)
            svm_i.set_params(C=16, tau=0.1)
            svm_i.run()

if __name__ == "__main__":

    X = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
    T = [1,-1,1,-1,1,-1]
    cv = CrossValidation(X, T, M=3)
    tr_set, val_set = cv.split_by_index(2)
    print "tr_set:", tr_set
    print "val_set:", val_set
    cv.do_cross_validation()
