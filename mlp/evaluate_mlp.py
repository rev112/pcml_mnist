import pickle
import sys
import scipy.io
import numpy as np

usage = '''Usage:
python evaluate_mlp.py <dataset_file>

where <dataset_file> should be a .mat file with entries 'TrainSet' and similar
'''
 
def run_testing(mat_fname):
    matFileContent = scipy.io.loadmat(mat_fname) # corresponding MAT file

    mlp = pickle.load(open('trained_network.dat', 'rb'))

    x_test = np.array(matFileContent['TestSet'].tolist())
    t_test = np.array(matFileContent['TestClass'].tolist())

    print "Test log error and accuracy:"
    print mlp.get_input_error(x_test, t_test), \
            mlp.get_accuracy(x_test, t_test), "%"

if __name__ == '__main__':
    lenarg = len(sys.argv)
    if lenarg == 2:
        mat_fname = sys.argv[1]
        run_testing(mat_fname)
    else:
        print >> sys.stderr, usage
        sys.exit(1)


