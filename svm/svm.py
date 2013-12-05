#!/usr/bin/env python2

import scipy as s
import helpers as h

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

        # Tau for Gaussian kernel (Warning: there are two different taus in documents!)
        self.tau = 0.01
        self.C = 0.01

        self.eps = 1e-08
        self.compute_kernel_matrix()
        return

    ### Move to separate module?
    def is_in_I_0(self, i):
        assert i in range(self.n), "Invalid index i"
        return (0 < self.alpha[i]) and (self.alpha[i] < self.C)

    def is_in_I_plus(self, i):
        assert i in range(self.n), "Invalid index i"
        return (self.T[i] == 1 and h.in_range(self.alpha[i], -self.eps, 0)) or \
               (self.T[i] == -1 and h.in_range(self.alpha[i], self.C, self.C + self.eps))

    def is_in_I_minus(self, i):
        assert i in range(self.n), "Invalid index i"
        return (self.T[i] == -1 and h.in_range(self.alpha[i], -self.eps, 0)) or \
               (self.T[i] == 1 and h.in_range(self.alpha[i], self.C, self.C + self.eps))

    def filter_alpha(self, f):
        return filter(f, range(self.n))

    ###

    def recompute_I_sets(self):
        # Initialize I_low and I_up
        # TODO how to find indices in a cool way?
        I_0 = self.filter_alpha(self.is_in_I_0)
        I_plus = self.filter_alpha(self.is_in_I_plus)
        I_minus = self.filter_alpha(self.is_in_I_minus)
        assert len(I_0) + len(I_plus) + len(I_minus) == self.n, "Invalid I_* sets"
        assert set(I_0 + I_plus + I_minus) == set(range(self.n))

        self.I_low = I_minus + I_0
        self.I_up = I_plus + I_0

    def initialize_run(self):
        self.f = -self.T
        self.alpha = s.zeros(self.n)
        self.recompute_I_sets()
        return

    def run(self):
        self.initialize_run()
        T = self.T
        K = self.K
        step_n = 1
        while(1):
            (i, j) = self.select_pair()
            if j == -1:
                break
            sig = T[i] * T[j]

            # 1. Computer L, H

            eta = K[(i,i)] + K[(j,j)] - 2*K[(i,j)]
            if eta > 1e-15:
                # 2. Compute the minimum along the direction of the constraint
                print 'lala'
            else:
                print 'lolo'
                # 3. Compute F_H, F_L

            # 4. Compute new alpha_i

            # 5. Update alpha vector

            # 6. Update f

            # 7. Update I_low, I_up

            break
            step_n += 1
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
        self.K = K
        return

    def select_pair(self):
        """Choose violated pair (see 1.2 from SVM doc)"""
        f = self.f
        I_up, I_low = self.I_up, self.I_low

        i_up = I_up[f[I_up].argmin()]
        i_low = I_low[f[I_low].argmax()]
        assert i_low != i_up, "Indices are equal!"

        # Check for optimality
        if f[i_low] <= f[i_up] + 2*self.tau:
            i_low = -1
            i_up = -1
        return (i_low, i_up)


if __name__ == "__main__":
    X = s.matrix([  [1,2],
                    [2,5],
                    [3,4] ])
    T = s.array([1, -1, 1])
    svm = SVM(3,2,X,T)
    svm.run()


