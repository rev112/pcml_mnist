#!/usr/bin/env python2

import cross_validation
import scipy.io
import sys
import os
import time

all_C = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12]

all_tau1 = [0.001, 0.002, 0.004, 0.008, 0.016]
all_tau2 = [0.032, 0.064, 0.128, 0.256, 0.512]

all_tau = all_tau1

d = scipy.io.loadmat('../mnist/4svm/mp_4-9_data_prepr_shuf.mat')

train_datapoints = d['Xtrain']
train_classes = d['Ytrain'].flatten()
test_datapoints = d['Xtest']
test_classes = d['Ytest'].flatten()

pt_n = 6000
print 'Dataset size:', len(train_datapoints)

dir_name = 'logs_out'

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

for tau in all_tau:
    pid = os.fork()
    if (pid == 0):
        # Child
        out_file = dir_name + '/full_' + str(tau)
        sys.stdout = open(out_file, 'w')
        print "Process started at:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "\n- - -\n"
        cv = cross_validation.CrossValidation(train_datapoints[:pt_n], train_classes[:pt_n], M=10)
        new_tau_list = [tau]
        cv.check_parameters(C_list=all_C, tau_list=new_tau_list)
        print "\n- - -\nProcess terminated at:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) 
        sys.exit(0)
    else:
        print 'tau:', tau, ', child PID:', pid
        time.sleep(1)

