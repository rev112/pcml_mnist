import numpy as np
import scipy.io
import sys

def generate(fname):
    deviation = 0.3
    trainData = 15

    nbPointsPerQuadrant = 20
    xQuadrant = [-1, 1, 1, -1]
    yQuadrant = [-1, -1, 1, 1]
    labels = [-1, 1, -1, 1]
    labels = map(lambda x: 'X' if x==1 else 'Y', labels)

    data = {'X': [], 'Y': []}
    for quad in xrange(len(xQuadrant)):
        points = np.random.normal(1, deviation, (nbPointsPerQuadrant, 2))
        points *= np.array([[xQuadrant[quad]], [yQuadrant[quad]]])
        data[labels[quad]] += points

    split_data = {}
    split_data['Xtrain'] = data['X'][:15]
    split_data['Xtest'] = data['X'][15:]
    split_data['Ytrain'] = data['Y'][:15]
    split_data['Ytest'] = data['Y'][15:]

    scipy.io.savemat(fname, split_data, oned_as='column')
    print "Finished generating XOR dataset"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: generate.py MATFILE')
        sys.exit(1)
    generate(sys.argv[1])
