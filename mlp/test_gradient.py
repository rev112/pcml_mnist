from mlp import Mlp
import sys
import numpy as np
import scipy.io

usage = '''Usage:
python test_gradient.py <dataset_file>

where <dataset_file> should be a .mat file with entries 'Xtrain' and 'Xtest'
(other entries allowed but not used)
'''

def learn_with_gradient_testing(fname):
    matFileContent = scipy.io.loadmat(fname) # corresponding MAT file
    x_train = np.array(matFileContent['Xtrain'].tolist())
    t_train = np.array(matFileContent['Ytrain'].tolist())

    d = x_train.shape[1]
    hidden_layers_list = [10]

    mlp = Mlp(hidden_layers_list, d, True)
    stopping_criterion = Mlp.BasicStoppingCriterion(0.05, 10)

    error_data = mlp.train_network(x_train, t_train, x_train, t_train,
            stopping_criterion)

    print "Error data:"
    error_data = map(lambda x: repr(x), error_data)
    print reduce(lambda x, y: x+'\n'+y, error_data)

    print "Log error:"
    print mlp.get_input_error(x_train, t_train)

    try:
        x_test = np.array(matFileContent['Xtest'].tolist())
        t_test = np.array(matFileContent['Ytest'].tolist())
        print "Test log error:"
        print mlp.get_input_error(x_test, t_test)
    except:
        pass

    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr, usage
        sys.exit(1)

    learn_with_gradient_testing(sys.argv[1])
