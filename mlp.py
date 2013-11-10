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
    """Common parent for layers"""


class OutputLayer(Layer):
    """Class for an output layer"""

    def __init__(self, d):
        self.d = d
        self.w = s.zeroes(neurons_num)
        self.b = 1.0

    def forward_step(self, x):
        """Return the value for the last layer (a number, not a vector)"""
        w = self.w
        b = self.b
        assert len(w) == d, "Invalid size of w"
        assert len(x) == d, "Invalid size of x"
        a = w.dot(x) + b
        assert type(a) == float
        return a

    def backward_step(x):
        """Return r error value for this layer"""
        return

    def update(r):
        """Update the parameters for this layer (w,b), given the error"""
        return


class HiddenLayer(Layer):
    """Class for one hidden layer"""

    def __init__(self, neurons_num, d):
        self.h = neurons_num
        self.d = d
        links = 2 * self.h
        self.w = s.matrix(s.zeros( (links, d) ))
        self.b = s.ones(links)


    def forward_step(self, x):
        """Return z_k = g(a_k, a_k+1) values for this layer (as a vector)"""
        assert len(x) == d, "Invalid size of input vector (x)"
        w = self.w
        b = self.b
        a_q = w.dot(x).A[0] + b
        assert len(a_q) == 2 * self.h, "Invalid size of a_q"

        # Apply transfer function
        # FIXME Is there a better way?
        z = map(lambda (x,y): g(x,y), zip(a_q[::2], a_q[1::2]))

        assert len(z) == self.h, "Invalid size of output vector (z)"
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
    l1 = HiddenLayer(neur_n, d)
    print l1.forward_step([1] * d)

