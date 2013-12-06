import pickle
import sys

def run_testing(mat_fname):
    matFileContent = scipy.io.loadmat(mat_fname) # corresponding MAT file
    data = np.array(matFileContent['Xtest'].tolist())

    mlp = pickle.load(open('trained_network.dat', 'rb'))

    # TODO: do testing


if __name__ == '__main__':
    lenarg = len(sys.argv)
    if lenarg == 2:
        mat_fname = sys.argv[1]
        run_testing(mat_fname)
    elif lenarg == 3:
        mat_fname = sys.argv[1]
        mlp_fname = sys.argv[2]
        run_testing(mat_fname, sys.argv[2])
    else:
        print >> sys.stderr, usage
        sys.exit(1)


