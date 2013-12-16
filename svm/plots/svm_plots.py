#!/usr/bin/env python2

import re
import sys
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#log_file = open('../train_6000_log.txt', 'r')
#file_str = log_file.read()


def find_steps():
    steps = re.findall(r'(?<=Step )\d+ ', file_str)
    steps = map(lambda x: int(x), steps)
    return steps


def plot_svm_criterion():
    # 1. SVM Criterion (F)
    all_F = re.findall(r'(?<=F:  )-?\d+\.\d+', file_str)
    all_F = map(lambda x: float(x), all_F)
    steps = find_steps()
    assert len(steps) == len(all_F)
    
    plt.plot(steps, all_F, color='r')
    plt.xlabel('Step number', fontsize=18)
    plt.ylabel(r'$\Phi(\alpha)$', fontsize=18)
    plt.title(r'SVM criterion $(\Phi(\alpha))$', fontsize=20)
    plt.show()

def plot_convergence_criterion():
    f_diff = []
    for m in re.finditer(r'(?<=f_up:) ( -?\d+\.\d+)( -?\d+\.\d+)', file_str):
        f_low, f_up = float(m.group(1)), float(m.group(2))
        f_diff.append(f_low - f_up)
    steps = find_steps()
    assert len(f_diff) == len(steps)

    plt.semilogy(steps, f_diff, color='b')
    #plt.plot(steps, f_diff, color='b')
    plt.xlabel('Step number', fontsize=18)
    plt.ylabel(r'Convergence criterion ($f_{low} - f_{up}$)', fontsize=18)
    plt.title(r'Convergence criterion ($f_{low} - f_{up}$)', fontsize=20)
    plt.show()
    

def bar_plot_svm_mlp():
    N = 2

    ind = np.arange(N)  # the x locations for the groups
    width = 0.3     # the width of the bars

    # Training set, test set

    mlp = (0.024, 0.013)
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, mlp, width, color='r')

    svm = (0.047, 0.042)
    rects2 = ax.bar(ind+width, svm, width, color='y')

    ax.set_ylabel('Zero/one error')
    ax.set_title('SVM and MLP comparison')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('Training set', 'Test set') )
    ax.legend( (rects1[0], rects2[0]), ('MLP', 'SVM') )

    plt.show()


#plot_svm_criterion()
#plot_convergence_criterion()

bar_plot_svm_mlp()
