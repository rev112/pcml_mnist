#!/bin/bash

#ARCHITECTURES='[10] [20] [50] [100] [5,10] [10,5]'
ARCHITECTURES='[100] [5,10] [10,5]'

for arch in $ARCHITECTURES; do
    echo "ARCHITECTURE ${arch}"
    echo ""
    for k in {1..2}; do
        python2 train_mlp.py ../mnist/mp_3-5_data_split.mat $arch 
        echo ""
    done
done
