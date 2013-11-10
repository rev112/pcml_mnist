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

