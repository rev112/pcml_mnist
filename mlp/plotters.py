import matplotlib.pyplot as plt

def plot_network_errors(network_errors, best_epoch,
        figure_name = 'plots/errors.png'):
    """Plot both train and validation errors"""
    epochs, training_errors, validation_errors = zip(*network_errors)

    print "MINIMUM VALIDATION ERROR:\t", \
            min(validation_errors)

    plt.plot(epochs, training_errors, 'b', label='Training error')
    plt.plot(epochs, validation_errors, 'g', label='Validation error')

    the_epoch = [best_epoch, best_epoch + 0.01]
    bar = [0.0, max(max(training_errors), max(validation_errors)) + 0.1]
    plt.plot(the_epoch, bar, 'r', label='Chosen MLP')
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Logistic error')
    plt.savefig(figure_name, bbox_inches = 0)


