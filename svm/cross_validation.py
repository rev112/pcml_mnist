#!/usr/bin/env python2

import scipy as s
import svm

class CrossValidation:

    def __init__(self, X, M=10):
        self.X = X
        self.n = len(X)
        self.M = M

    def split_by_index(self, i):
        """Return a tuple of training set (without part i) 
        and validation set (part i)"""
        assert 1 <= i and i <= self.M, "Invalid part number"
        part_size = self.n / self.M
        lbound = part_size * (i-1)
        hbound = part_size * i
        training_set = X[:lbound] + X[hbound:]
        validation_set = X[lbound:hbound]
        return (training_set, validation_set)

if __name__ == "__main__":

    X = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
    cv = CrossValidation(X, M=3)
    tr_set, val_set = cv.split_by_index(2)
    print "tr_set:", tr_set
    print "val_set:", val_set
