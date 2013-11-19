#!/usr/bin/env python2

import numpy as np

def sig(v):
    """Compute the sigma function: 1 / (1 + e^(-x))
    @param v - ndarray
    """
    return 1.0 / (1.0 + np.exp(-v))

def g_scalar(a1, a2):
    """Compute the transfer function (g): a1*sig(a2)"""
    return 1.0 * a1 * sig(a2)

def g(a1, a2):
    """vectorized g funtion, a1 and a2 are ndarrays
    g = a1 * sig(a2)
    """
    return a1 * sig(a2)

def g_der_1(a1, a2):
    """Compute the partial derivative of g wrt first argument"""
    return sig(a2)

def g_der_2(a1, a2):
    """Compute the partial derivative of g wrt second argument"""
    return a1 * (1.0 - sig(a2)) * sig(a2)

