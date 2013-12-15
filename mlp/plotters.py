import matplotlib.pyplot as plt

def plot_network_errors(network_errors, figure_name = 'plots/errors.png'):
    """Plot both train and validation errors"""
    epochs, training_errors, validation_errors = zip(*network_errors)

    plt.plot(epochs, training_errors, 'r', label='Training error')
    plt.plot(epochs, validation_errors, 'b', label='Validation error')
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Logistic error')
    plt.savefig(figure_name, bbox_inches = 0)

