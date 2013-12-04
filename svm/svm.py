#!/usr/bin/env python2

import scipy as s

class SVM:
    """SVM class

        n - number of datapoints
        d - size of one datapoint
        X - matrix (of shape n * d) of all datapoints
        T - vector (of size n) of datapoint classes (+1 or -1)
    """

    def __init__(self, n, d, X, T):
        self.n = n
        self.d = d
        X = s.matrix(X)
        T = s.array(T)
        assert X.shape == (n,d), "Invalid shape of data matrix X"
        assert len(T) == n, "Invalid size of class vector T"
        self.X = X
        self.T = T

        self.tau = 1e-08
        self.C = 1

        self.compute_kernel_matrix()
        return

    def initialize_run(self):
        self.f = -self.T
        self.alpha = s.ones(self.n)

        # Initialize I_low and I_up

        # TODO how to find indices in a cool way?
        I_0 = set()
        I_plus = set()
        I_minus = set()
        self.I_low = I_plus.union(I_0)
        self.I_up = I_minus.union(I_0)
        return

    def run(self):
        self.initialize_run()
        return

    def compute_kernel_matrix(self):
        """Compute kernel matrix (see 2.1 from SVM doc)"""
        n = self.n

        # 1. compute d
        xxt = X * X.transpose()
        d = s.diag(xxt)
        d = s.matrix(d).transpose()
        print 'd', d

        # 2. compute A
        ones = s.matrix(s.ones(n)).transpose()
        A = (1.0/2) * d * ones.transpose()
        A += (1.0/2) * ones * d.transpose()
        A -= xxt
        print 'A', A

        # 3. compute K with Gaussian kernel
        f = s.vectorize(lambda a : s.exp(-self.tau*a))
        K = f(A)
        assert K.shape == (n,n), "Invalid shape of kernel matrix"
        print 'K', K
        return

    def select_pair(self):
        """Choose violated pair (see 1.2 from SVM doc)"""
        i_low = True
        i_up = False
        return [i_low, i_up]


if __name__ == "__main__":
    X = s.matrix([  [1,2],
                    [2,3],
                    [3,4] ])
    T = s.array([1, -1, 1])
    svm = SVM(3,2,X,T)
    svm.run()


