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

    def __init__(self, X, T):
        self.n = len(X)
        self.d = len(s.array(X)[0])
        X = s.matrix(X)
        T = s.array(T)
        assert X.shape == (self.n, self.d), "Invalid shape of data matrix X"
        assert len(T) == self.n, "Invalid size of class vector T"
        self.X = X
        self.T = T

        # Tau for Gaussian kernel (Warning: there are two different taus in documents!)
        self.set_params()
        assert self.C > 0, "Invalid C"

        self.eps = 1e-08
        self.compute_kernel_matrix()

        self.b = 0
        return

    def set_params(self, C=0.01, tau=0.01):
        self.C = C
        self.tau = tau

    ### Move to separate module?
    ### TODO optimize! Seems that these functions create a bottleneck
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

    def compute_I_sets(self):
        # Initialize I_low and I_up
        # TODO how to find indices in a cool way?
        I_0 = self.filter_alpha(self.is_in_I_0)
        I_plus = self.filter_alpha(self.is_in_I_plus)
        I_minus = self.filter_alpha(self.is_in_I_minus)
        assert len(I_0) + len(I_plus) + len(I_minus) == self.n, "Invalid I_* sets"
        assert set(I_0 + I_plus + I_minus) == set(range(self.n))
        # print "I_0, I_plus, I_minus:", I_0, I_plus, I_minus

        self.I_low = I_minus + I_0
        self.I_up = I_plus + I_0
        self.I_0 = I_0
        assert len(self.I_low) != 0, "Only +1 classes?"
        assert len(self.I_up) != 0, "Only -1 classes?"

    def update_element_I_set(self, i):
        I_0 = self.I_0
        I_low = self.I_low
        I_up = self.I_up
        # Remove from sets
        if i in I_0:
            I_0.remove(i)
            I_low.remove(i)
            I_up.remove(i)
        elif i in I_low:
            I_low.remove(i)
        elif i in I_up:
            I_up.remove(i)

        # Add to sets
        if self.is_in_I_0(i):
            I_0.append(i)
            I_up.append(i)
            I_low.append(i)
        elif self.is_in_I_plus(i):
            I_up.append(i)
        elif self.is_in_I_minus(i):
            I_low.append(i)

    def update_I_sets(self, i, j):
        self.update_element_I_set(i)
        self.update_element_I_set(j)

    def initialize_run(self):
        self.f = -self.T * 1.0
        self.alpha = s.zeros(self.n)
        self.compute_I_sets()

    def run(self):
        self.initialize_run()
        T = self.T
        K = self.K
        step_n = 0
        print "\n>>> New run " + ">"*20
        print "C =", self.C, ", tau =", self.tau
        while(1):
            outstr = ''
            #cur_F = self.compute_F()
            #print "Step", step_n , ", F: ", cur_F
            outstr += "Step " + str(step_n) + "\n"
            #print "Alphas:", self.alpha
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

            alpha_i_new = self.adjust_alpha(alpha_i_new)
            alpha_j_new = self.adjust_alpha(alpha_j_new)
            #outstr += "new i:" + str(alpha_i_new) + ", new j:" + str(alpha_j_new) + "\n"

            # 5. Update alpha vector
            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new

            # 6. Update f
            self.f += T[i] * (alpha_i_new - alpha_i) * s.array(K[i])[0]
            self.f += T[j] * (alpha_j_new - alpha_j) * s.array(K[j])[0]

            # 7. Update I_low, I_up
            self.update_I_sets(i, j)

            if step_n % 10 == 0:
                print outstr
            step_n += 1
        self.recompute_b()
        print "b:", self.b
        return

    def compute_F(self):
        """Compute target function F(alpha). For debug purposes!!!"""
        # 1. Check constraints
        assert s.all(self.alpha >= 0)
        assert s.all(self.alpha <= self.C)
        alpha_t = self.alpha * self.T
        assert h.in_range(alpha_t.sum(), -self.eps, self.eps)

        # 2. Compute F
        sum1 = alpha_t.dot(self.K).dot(alpha_t)
        sum2 = self.alpha.sum()
        F = 0.5 * sum1 - sum2
        assert F.shape == (1,1)
        F = F.tolist()[0][0]
        return F

    def recompute_b(self):
        if len(self.I_0) == 0:
            self.b = 0
            return
        b = 0
        for i in self.I_0:
            # y_i_tilda = sum_j (alpha_j * T_j * K(x_j, x_i))
            y_i_tilda = (self.alpha * self.T).dot(s.array(self.K[i])[0])
            # We change the sign, because in our case y = sum(a t K) - b
            b += -(self.T[i] - y_i_tilda)
        self.b = 1.0 * b / len(self.I_0)


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

        return (F_L, F_H)

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
        return (L, H)

    def adjust_alpha(self, alpha_i):
        C = self.C
        if h.in_range(alpha_i, -self.eps, self.eps):
            return 0
        if h.in_range(alpha_i, C - self.eps, C + self.eps):
            return C
        return alpha_i

    def compute_kernel_matrix(self):
        """Compute kernel matrix (see 2.1 from SVM doc)"""
        print "Computing kernel matrix..."
        n = self.n
        X = self.X
        tau = self.tau

        # 1. compute d
        xxt = X * X.transpose()
        d = s.diag(xxt)
        d = s.matrix(d).transpose()

        # 2. compute A
        ones = s.matrix(s.ones(n))
        A = 0.5 * d * ones
        A += 0.5 * ones.transpose() * d.transpose()
        A -= xxt

        # 3. compute K with Gaussian kernel
        A = -tau*A
        K = s.exp(A)
        assert K.shape == (n,n), "Invalid shape of kernel matrix"
        self.K = K
        print "Finished computing kernel matrix."
        return

    def select_pair(self):
        """Choose violated pair (see 1.2 from SVM doc)"""
        f = self.f
        I_up, I_low = self.I_up, self.I_low

        i_up = I_up[f[I_up].argmin()]
        i_low = I_low[f[I_low].argmax()]

        # Check for optimality
        #print "f_low, f_up: ", f[i_low], f[i_up]
        if f[i_low] <= f[i_up] + 2*self.eps:
            i_low = -1
            i_up = -1
        assert i_low == -1 or i_low != i_up, "Indices are equal!"
        return (i_low, i_up)

    def get_output_2d(self, xs):
        """Compute outputs for the array of datapoints"""
        X = self.X
        n = self.n
        m = len(xs)
        Y = s.matrix(xs)

        yyt = Y * Y.transpose()
        d_y = s.diag(yyt)
        d_y = s.matrix(d_y).transpose()
        ones_y = s.matrix(s.ones(n))

        xt = X.transpose()
        xxt = X * xt
        d_x = s.diag(xxt)
        d_x = s.matrix(d_x)
        ones_x = s.matrix(s.ones(m)).transpose()

        # A is (m, n) matrix
        A = 0.5 * (d_y * ones_y + ones_x * d_x)
        A -= Y * xt

        K_val = s.exp(-self.tau*A)
        alpha_t = self.alpha * self.T
        y_out = K_val.dot(alpha_t) - s.matrix([self.b] * m)

        return s.asarray(y_out)[0]

    def get_output(self, x_new):
        """Compute the discriminant function (y) for a given datapoint"""
        X = self.X
        x_new = s.array(x_new)
        assert len(x_new) == self.d
        alpha_t = self.alpha * self.T
        assert len(alpha_t) == self.n

        K_vect = map(lambda x_i: h.gaussian_kernel_function(x_new, x_i, self.tau),
                                 s.array(X))
        K_vect = s.array(K_vect)
        y = alpha_t.dot(K_vect) - self.b
        return y

    def classify_output(self, y):
        """Classify SVM output"""
        cl = int(s.sign(y))
        if cl == 0:
            import random
            print "Warning, class = 0, assigning label 1..."
            cl = 1
        return cl

    def classify(self, x_new):
        """Classify a new datapoint"""
        y = self.get_output(x_new)
        return self.classify_output(y)

if __name__ == "__main__":
    X = s.matrix([  [1,2],
                    [2,5],
                    [3,4] ])
    T = s.array([1, -1, 1])
    svm = SVM(X,T)
    svm.set_params(C=16, tau=0.1)
    svm.run()
    print 'SVM alphas:',svm.alpha
    print svm.classify([1,2])
    print svm.classify([2,5])
    print svm.classify([3,4])
    print '1d', svm.get_output([1,2])
    print '1d', svm.get_output([2,5])
    print '2d', svm.get_output_2d([[1,2],[2,5]])


