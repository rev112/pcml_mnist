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
            L, H = self.compute_L_H(i, j)

            eta = K[(i,i)] + K[(j,j)] - 2*K[(i,j)]
            if eta > 1e-15:
                # 2. Compute the minimum along the direction of the constraint, clip
                alpha_j_new = self.compute_min_and_clip(i, j, L, H, eta)
            else:
                # 3. Compute F_H, F_L
                F_L, F_H = self.compute_F_LH(i, j, L, H)
                if F_L > F_H:
                    alpha_j_new = H
                else:
                    alpha_j_new = L

            # 4. Compute new alpha_i
            alpha_i = self.alpha[i]
            alpha_j = self.alpha[j]
            alpha_i_new = alpha_i + sig * (alpha_j - alpha_j_new)

            # 5. Update alpha vector
            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new

            # 6. Update f
            self.f += T[i] * (alpha_i_new - alpha_i) * s.array(K[i])[0]
            self.f += T[j] * (alpha_j_new - alpha_j) * s.array(K[j])[0]

            # 7. Update I_low, I_up
            self.recompute_I_sets()

            step_n += 1
        return

    def compute_F_LH(self, i, j, L, H):
        K = self.K
        T = self.T
        f = self.f
        alpha = self.alpha
        sig = T[i] * T[j]
        w = alpha[i] + sig * alpha[j]
        v_i = f[i] + T[i] - alpha[i]*T[i]*K[(i,i)] - alpha[j]*T[j]*K[(i,j)]
        v_j = f[j] + T[j] - alpha[i]*T[i]*K[(i,j)] - alpha[j]*T[j]*K[(j,j)]

        L_i = w - sig * L
        F_L = 0.5 * (K[(i,i)] * L_i * L_i + K[(j,j)] * L * L)
        F_L += sig * K[(i,j)] * L_i * L
        F_L += T[i] * L_i * v_i + T[j] * L * v_j - L_i - L

        H_i = w - sig * H
        F_H = 0.5 * (K[(i,i)] * H_i * H_i + K[(j,j)] * H * H)
        F_H += sig * K[(i,j)] * H_i * H
        F_H += T[i] * H_i * v_i + T[j] * H * v_j - H_i - H

        return [F_L, F_H]

    def compute_min_and_clip(self, i, j, L, H, eta):
        f = self.f
        alpha_j_unc = self.alpha[j] + 1.0 * self.T[j] * (f[i] - f[j]) / eta
        alpha_j = alpha_j_unc
        if alpha_j < L:
            alpha_j = L
        elif alpha_j_unc > H:
            alpha_j = H
        return alpha_j

    def compute_L_H(self, i, j):
        """Compute L and H"""
        T = self.T
        C = self.C
        sig = T[i] * T[j]
        sig_w = self.alpha[j] + sig * self.alpha[i]

        L = max(0,      sig_w - C * h.indicator(sig == 1))
        H = min(self.C, sig_w + C * h.indicator(sig == -1))
        return [L, H]

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


