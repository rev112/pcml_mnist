#!/usr/bin/env python2

import scipy as s

class Layer:
    """Class for one hidden layer"""

    def __init__(self, neurons_num, d):
        self.h1 = neurons_num
        self.d = d
        links = 2 * self.h1
        self.w = s.matrix(s.zeros((links, d)))
        self.b = s.ones(links)

    def forward_step(self, x):
        """Return a_k values for this layer"""
        w = self.w
        b = self.b
        return w.dot(x).A[0] + b

    def backward_step(x):
        """Return r_k error values for this layer"""
        return

    def update(r):
        """Updates the parameters for this layer, given the errors"""
        return

    def sig(v):
        """Compute the sigma function: 1 / (1 + e^(-x))"""
        return 1.0 / (1 + s.exp(-v))

    def g(a1, a2):
        """Compute the transfer function (g): a1*sig(a2)"""
        return a1 * sigma(a2)


if __name__ == "__main__":
    d = 5
    neur_n = 1
    l1 = Layer(neur_n, d)
    print l1.forward_step([1] * d)

