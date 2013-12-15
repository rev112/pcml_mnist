#!/usr/bin/env python2

import scipy.io
import scipy as s
import random
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
import mlp


def plot_network_errors(network_errors):
    """Plot both train and validation errors"""
    epochs, training_errors, validation_errors = zip(*network_errors)

    plt.plot(epochs, training_errors, 'r', label='Training error')
    plt.plot(epochs, validation_errors, 'b', label='Validation error')
    plt.legend(loc=1)
    plt.xlabel('Epoch number')
    plt.ylabel('Logistic error')
    plt.show()


def shuffle_2d(a, b):
    """Shuffle 2 arrays simultaneously"""
    rng_st = s.random.get_state()
    s.random.shuffle(a)
    s.random.set_state(rng_st)
    s.random.shuffle(b)


# Test 
d = scipy.io.loadmat('../../mnist/mp_3-5_data_split.mat')
train_datapoints = d['TrainSet']
train_classes = d['TrainClass'].flatten()
valid_datapoints = d['ValidSet']
valid_classes = d['ValidClass'].flatten()
test_datapoints = d['TestSet']
test_classes = d['TestClass']

dtp = 400

# Shuffle datapoints
shuffle_2d(train_datapoints, train_classes)
shuffle_2d(valid_datapoints, valid_classes)
train_datapoints = train_datapoints[:dtp]
train_classes = train_classes[:dtp]
valid_datapoints = valid_datapoints[:dtp]
valid_classes = valid_classes[:dtp]

# Use the part of them to train the network
mlp = mlp.Mlp(hidden_layers_list = [2], d = len(train_datapoints[0]))

stop_crit = mlp.BasicStoppingCriterion(0.0000001, 10)
#stop_crit = mlp.EarlyStoppingCriterion()
res = mlp.train_network(train_datapoints, train_classes, valid_datapoints, valid_classes, stop_crit)
print res
plot_network_errors(res)

test_classes = test_classes.flatten()
print test_classes

classified_correctly = 0
total_len = len(test_datapoints)
for i in xrange(len(test_datapoints)):
    dp = test_datapoints[i]
    cl = test_classes[i]
    out_cl = mlp.classify(dp)
    if cl == out_cl:
        classified_correctly += 1

print classified_correctly, total_len, 100.0*classified_correctly/total_len

