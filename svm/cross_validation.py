#!/usr/bin/env python2

import scipy as s
import svm
from itertools import product

class CrossValidation:
    """Class for performing a cross-validation

        X - whole dataset (datapoints)
        T - datapoints' classes
        n - total number of datpoints
        d - size of one datapoint
        M - we're going to perform M-fold cross-validation
    """

    def __init__(self, X, T, M=10):
        self.X = s.array(X)
        self.T = s.array(T)
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
        dtp_training = s.concatenate((X[:lbound],X[hbound:]))
        cl_training = s.concatenate((T[:lbound], T[hbound:]))
        dtp_validation = X[lbound:hbound]
        cl_validation = T[lbound:hbound]
        training_set = {'dtp': dtp_training, 'cl': cl_training}
        validation_set = {'dtp': dtp_validation, 'cl': cl_validation}
        return (training_set, validation_set)

    def find_init_values(self):
        C_cur = 2**(-4)
        C_res = {}
        tau_start = 0.1
        print "Fixed tau, different C"
        for i in xrange(10):
            estimator = self.do_cross_validation(C_cur, tau_start)
            C_res[C_cur] = estimator
            C_cur *= 2
        print C_res

        C_start = 0.1
        tau_cur = 2**(-4)
        tau_res = {}
        print "Fixed C, different tau"
        for i in xrange(10):
            estimator = self.do_cross_validation(C_start, tau_cur)
            tau_res[tau_cur] = estimator
            tau_cur *= 2

        print tau_res

    def check_parameters(self, C_list=[], tau_list=[]):
        C_init_value = 0.1
        tau_init_value = 0.001
        if not C_list: C_list = [C_init_value * 2**i for i in range(10)]
        if not tau_list: tau_list = [tau_init_value * 2**i for i in range(10)]
        print 'C_list', C_list
        print 'tau_list', tau_list

        res = {}
        for (C, tau) in product(C_list, tau_list):
            estimator = self.do_cross_validation(C, tau)
            estimator = round(estimator, 3)
            res[(C,tau)] = estimator
            print "\nCHECK_PARAMETERS RESULTS: C =", C, ", tau =", tau, ", CV estimator:", estimator
        print "\n- - -\nAll combinations:", res
        min_key = min(res, key=res.get)
        print 'Values (C,tau) with minimum estimator value:', min_key


    def do_cross_validation(self, C, tau):
        """Perform M-fold cross-validation and return estimator"""
        svm_list = []
        cv_estimator = 0.0
        print "\n" + "# " * 30
        print ">>> Started cross validation for C =", C, ",tau =", tau
        for i in range(1, self.M + 1):
            print ">>> Run number %u (of %u)" % (i, self.M)
            print "> Total dataset size:", len(self.X)
            tr_set_i, val_set_i = self.split_by_index(i)
            tr_set_size = len(tr_set_i['dtp'])
            svm_i = svm.SVM(tr_set_i['dtp'], tr_set_i['cl'])
            svm_list.append(svm_i)
            svm_i.set_params(C, tau)
            svm_i.run()
            estimator_i = self.compute_estimator(svm_i, val_set_i)
            cv_estimator += estimator_i
        cv_estimator = 1.0 * cv_estimator / self.M
        return cv_estimator

    def compute_estimator(self, svm, validation_set):
        """Compute least-square estimator for validation set"""
        val_set_dp = validation_set['dtp']
        val_set_cl = validation_set['cl']
        estimator = 0.0
        validation_size = len(val_set_dp)
        print "Computing estimator with validation part..."

        val_output = svm.get_output_2d(val_set_dp)

        # Compute estimator
        diff = val_set_cl - val_output
        estimator = diff.dot(diff).sum()
        # See p. 183
        estimator *= 1.0 * self.M / self.n

        classify_vect = s.vectorize(svm.classify_output)
        output_classes = classify_vect(val_output)

        diff_classes = output_classes - val_set_cl
        errors = s.count_nonzero(diff_classes)
        classified_correctly = validation_size - errors

        print "Classified correctly: %u/%u (%.2f%%)" % \
              (classified_correctly, validation_size,
               100.0 * classified_correctly / validation_size)
        return estimator

if __name__ == "__main__":

    X = [[1,2],[1,-1],[2,2],[2,-1],[3,3],[3,-1]]
    T = [1,-1,1,-1,1,-1]
    cv = CrossValidation(X, T, M=2)
    tr_set, val_set = cv.split_by_index(2)
    print "tr_set:", tr_set
    print "val_set:", val_set
    cv.check_parameters()
