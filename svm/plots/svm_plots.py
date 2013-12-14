#!/usr/bin/env python2

import matplotlib.pyplot as plt
import re
import sys


log_file = open('../train_6000_log.txt', 'r')
file_str = log_file.read()


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
    plt.xlabel('Step number')
    plt.ylabel('SVM criterion (Phi)')
    plt.title('SVM criterion (Phi)')
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
    plt.xlabel('Step number')
    plt.ylabel('Convergence criterion (f_low - f_up)')
    plt.title('Convergence criterion (f_low - f_up)')
    plt.show()
    


plot_svm_criterion()
#plot_convergence_criterion()
