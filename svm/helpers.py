#!/usr/bin/env python2

import scipy as s

def in_range(number, down, up):
    return (down <= number) and (number <= up)

def indicator(cond):
    """Compute indicator function: cond ? 1 : 0"""
    res = 1 if cond else 0
    return res


def gaussian_kernel_function(x1, x2, tau):
    x1 = s.array(x1)
    x2 = s.array(x2)
    assert len(x1) == len(x2)
    diff = x1 - x2
    norm = diff.dot(diff)
    res = s.exp(-0.5 * tau * norm)
    return res
