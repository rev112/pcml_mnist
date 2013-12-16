#!/bin/bash

#ARCHITECTURES='[10] [20] [50] [100] [5,10] [10,5]'
ARCHITECTURES='[50] [100] [5,10] [10,5]'

for arch in $ARCHITECTURES; do
    for k in {0..3}; do
        python2 train_mlp.py ../mnist/mp_3-5_data_split.mat $arch 
    done
done
