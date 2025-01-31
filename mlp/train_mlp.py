from plotters import plot_network_errors
import defaults
import pickle
from mlp import Mlp
import sys
import numpy as np
import scipy.io

usage = '''Usage:
python train_mlp.py <dataset_file> [<network architecture>]

where <dataset_file> should be a .mat file with entries 'TrainSet' and similar
and <network architecture> is optional parameter for network architecture
(please put quotes, or no whitespace when describing architecture)
'''

def learn(argv):
    fname = argv[1]
    if len(argv) == 3:
        hidden_layers_list = eval(argv[2])
    else:
        hidden_layers_list = [10]

    matFileContent = scipy.io.loadmat(fname) # corresponding MAT file
    x_train = np.array(matFileContent['TrainSet'].tolist())
    t_train = np.array(matFileContent['TrainClass'].tolist())

    x_valid = np.array(matFileContent['ValidSet'].tolist())
    t_valid = np.array(matFileContent['ValidClass'].tolist())

    d = x_train.shape[1]

    mlp = Mlp(hidden_layers_list, d)
    stopping_criterion = Mlp.EarlyStoppingCriterion(5, 1e-5)
    #stopping_criterion = Mlp.BasicStoppingCriterion(0.001, 100)

    (error_data, best_epoch) = mlp.train_network(x_train, t_train, 
            x_valid, t_valid, stopping_criterion)

    lrate = defaults.LEARNING_RATE_DEFAULT
    mterm = defaults.MOMENTUM_TERM_DEFAULT 
    terms = str(lrate)[2:]+"_"+str(mterm)[2:]
    arch_desc = reduce(lambda x, y:str(x)+"_"+str(y), 
            hidden_layers_list, "")
    plt_file = 'plots/errors_' + terms + arch_desc + '.png'
    plot_network_errors(error_data, best_epoch, plt_file)

    print "Train log error and accuracy:"
    print mlp.get_input_error(x_train, t_train), \
            mlp.get_accuracy(x_train, t_train), "%"
    print "Valid log error and accuracy:"
    print mlp.get_input_error(x_valid, t_valid), \
            mlp.get_accuracy(x_valid, t_valid), "%"

    x_test = np.array(matFileContent['TestSet'].tolist())
    t_test = np.array(matFileContent['TestClass'].tolist())
    print "Test log error and accuracy:"
    print mlp.get_input_error(x_test, t_test), \
            mlp.get_accuracy(x_test, t_test), "%"

    pickle.dump(mlp, open('trained_network.dat', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print >> sys.stderr, usage
        sys.exit(1)

    learn(sys.argv)
