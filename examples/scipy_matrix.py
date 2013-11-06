#!/usr/bin/env python

import scipy as s

def are_matrix_equal(m1, m2):
  return s.equal(m1, m2).all()

# Test matrices multiplication

m1 = s.matrix([[1, 2], [3, 4]])
m2 = s.matrix("0 1; 1 0")
m3 = s.matrix([[2, 1], [4, 3]])

print "m1 = ", m1
print "m2 = ", m2

print are_matrix_equal(m1*m2, m3)

# Check the matrix size
m4 = s.matrix([[1,2],[3,4],[5,6]])
height, width = m4.shape
print (height, width) == (3,2)
print height*width == m4.size


# Multiply matrix and vector
m1 = s.matrix("1 2; 3 4")
x1 = s.array([5,6])
r1 = m1.dot(x1) # works, but returns a matrix
a1 = r1.A[0]    # now it's a vector
print s.array_equal(a1, s.array([17,39]))

