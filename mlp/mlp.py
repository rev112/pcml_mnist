#!/usr/bin/env python2

import scipy as s
import functions as func

class Layer:
    """Common parent for layers"""


class OutputLayer(Layer):
    """Class for an output layer"""

    def __init__(self, d):
        self.d = d
        self.w = s.ones(d)
        self.b = 1.0

    def forward_step(self, x):
        """Return the value for the last layer (a number, not a vector)"""
        w = self.w
        b = self.b
        d = self.d
        assert len(w) == d, "Invalid size of weight vector (w)"
        assert len(x) == d, "Invalid size of input vector (x)"
        a = w.dot(x) + b
        assert type(a) == s.float64
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
        w = self.w
        b = self.b
        d = self.d
        assert len(x) == d, "Invalid size of input vector (x)"
        a_q = w.dot(x).A[0] + b
        assert len(a_q) == 2 * self.h, "Invalid size of a_q"

        # Apply transfer function
        # FIXME Is there a better way?
        z = map(lambda (x,y): func.g(x,y), zip(a_q[::2], a_q[1::2]))

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
    l2 = OutputLayer(neur_n)
    print l2.forward_step([1] * neur_n)

