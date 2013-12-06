#!/usr/bin/env python2

def in_range(number, down, up):
    return (down <= number) and (number <= up)



def indicator(cond):
    """Compute indicator function: cond ? 1 : 0"""
    res = 1 if cond else 0
    return res
