#!/usr/bin/env python

from numpy import *
import scipy.io
import sys

if len(sys.argv) < 2:
  sys.stderr.write("Usage: ./load_mnist.py MATFILE")
  sys.exit(1);

d = scipy.io.loadmat(sys.argv[1]) # corresponding MAT file
keys = filter(lambda x: x[:2] != '__', d.keys())
keys.sort()

for k in keys:
  data = d[k]

  print k, 'set: ',data.shape[0],'datapoints'
