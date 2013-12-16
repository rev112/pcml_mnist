Support Vector Machines
=======================

This folder contains the code for SVM.

Important files:

`svm.py` - contains main **code for SVM**

`cross_validation.py` - contains **code for cross-validation**, which uses `svm.py`

`helpers.py` - few helping functions (some math)

`train_mnist.py` - file with **main running code** for training SVM on a MNIST dataset

`check_params_wrapper.py` - contains a wrapper for evaluating CV estimators for different combinations of C and tau. Warning, this script will spawn several child processes and computations can take quite a long time!

`plots/` - directory with plots and scripts for building them

`results/` - directory with results/log files
