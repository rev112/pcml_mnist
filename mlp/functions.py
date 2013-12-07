#!/usr/bin/env python2

import numpy as np

def sig(v):
    """Compute the sigma function: 1 / (1 + e^(-x))
    @param v - ndarray or float
    """
    return 1.0 / (1.0 + np.exp(-v))

def g_scalar(a1, a2):
    """Compute the transfer function (g): a1*sig(a2)"""
    return 1.0 * a1 * sig(a2)

def g(a1, a2):
    """Vectorized g function, a1 and a2 are ndarrays
    g = a1 * sig(a2)
    """
    assert len(a1) == len(a2), "Inconsistent vector lengths"
    return a1 * sig(a2)

def g_der_1(a1, a2):
    """Compute the partial derivative of g wrt first argument"""
    return sig(a2)

def g_der_2(a1, a2):
    """Compute the partial derivative of g wrt second argument"""
    return a1 * (1.0 - sig(a2)) * sig(a2)


# Helpers for matrices and arrays

def get_first_row(matrix):
    return np.asarray(matrix)[0]

def duplicate_columns(m):
    # Example: [[1,2], [3,4]] -> [[1,1,2,2], [3,3,4,4]]

    # TODO really dirty way, rewrite it
    m = np.matrix(m).tolist()
    extend_row = lambda row: np.array(map(lambda x: [x,x], row)).flatten()
    result_matrix = np.matrix(map(extend_row, m))
    return result_matrix


def get_random_direction(d):
    """Computes random direction vector of length d"""

    direction = np.random.normal(size=d)
    
#    success = False
#    while not success:
#        print "trying", d
#        direction = np.random.uniform(-1, 1, d)
#        r_squared = np.sum(direction**2)
#        if 0 < r_squared and r_squared <= 1.0:
#            success = True
            
    # return normalized
    #return direction / np.sqrt(r_squared)
    return direction / np.linalg.norm(direction)


def computeGradientApproximation(mlp, lx, lt, eps = 1e-4):
    """Computes an approximation (symmetric finite difference) of error gradient
    """
    d = get_random_direction(mlp.get_dimension())
    
    # TODO: get huge w vector

    # TODO: create MLPs with modified weights

    # TODO: get errors E(w + eps*d) and E(w - eps*d)

    return (Eplus - Eminus) / 2 / eps


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def testplot_random_directions():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    d = 3
    for k in xrange(1000):
        dir = get_random_direction(d)
        ax.scatter(dir[0], dir[1], dir[2])
    plt.show()

if __name__ == "__main__":
    testplot_random_directions()

