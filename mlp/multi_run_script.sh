#!/bin/bash

ARCHITECTURES='[10,5] [10, 10]'

for arch in $ARCHITECTURES; do
    echo "ARCHITECTURE ${arch}"
    echo ""
    for k in {1..1}; do
        python2 train_mlp.py ../mnist/mp_4-9_data_split.mat $arch 
        echo ""
    done
done
