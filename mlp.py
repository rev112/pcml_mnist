#!/usr/bin/env python2

import scipy as s

def sig(v):
    """Compute the sigma function: 1 / (1 + e^(-x))"""
    return 1.0 / (1 + s.exp(-v))

def g(a1, a2):
    """Compute the transfer function (g): a1*sig(a2)"""
    return 1.0 * a1 * sig(a2)

def g_der_1(a1, a2):
    """Compute the partial derivative of g wrt first argument"""
    return sig(a2)

def g_der_2(a1, a2):
    """Compute the partial derivative of g wrt second argument"""
    return a1 * (1 - sig(a2)) * sig(a2)

class Layer:
    """Class for one hidden layer"""

    def __init__(self, neurons_num, d):
        self.h1 = neurons_num
        self.d = d
        links = 2 * self.h1
        self.w = s.matrix(s.zeros((links, d)))
        self.b = s.ones(links)

    def forward_step(self, x):
        """Return z_k = g(a_k, a_k+1) values for this layer (as a vector)"""
        assert len(x) == d, "Invalid size of input vector (x)"
        w = self.w
        b = self.b
        a_q = w.dot(x).A[0] + b
        assert len(a_q) == 2 * self.h1, "Invalid size of a_q"

        # Apply transfer function
        # FIXME Is there a better way?
        z = map(lambda (x,y): g(x,y), zip(a_q[::2], a_q[1::2]))

        assert len(z) == self.h1, "Invalid size of output vector (z)"
        return z

    def backward_step(x):
        """Return r_k error values for this layer"""
        return

    def update(r):
        """Update the parameters for this layer, given the errors"""
        return

if __name__ == "__main__":
    d = 5
    neur_n = 2
    l1 = Layer(neur_n, d)
    print l1.forward_step([1] * d)

