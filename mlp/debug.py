#!/usr/bin/env python2

import pprint

pp = pprint.PrettyPrinter(indent=4)

def prt(*args):
    s = ''
    for arg in args:
        formatted = pp.pformat(arg)
        if type(arg) == str:
            formatted = formatted[1:-1]
        s = s + formatted + ' '
    print s
