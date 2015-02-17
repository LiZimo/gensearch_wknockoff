#!/bin/bash


SAMPLES=Iris_knockoff/data/100uniform1to5.csv
LABELS=data/iris_labels.csv
N=2
STEPS=20
K=5
METHOD=2
SPLITNUM=100
# This must be set as a directory/prefix
OUTPUT_PREFIX=output/Iris_knockoff

source python_wrapper.sh

which swift | grep "Swift 0.95 RC6"
if [[ "$?" != "0" ]]
then
    echo "ERROR: Must run script with Swift 0.95 RC6 or higher"
fi

echo "Cleaning previous results"
rm -rf $OUTPUT_PREFIX* &> /dev/null
rm -rf results
rm -rf tes

python preprocess.py -i $SAMPLES -l $LABELS -n $N -s $STEPS -k $K -m $METHOD -x $SPLITNUM -f $OUTPUT_PREFIX

swift gen_search.swift -n=$N \
    -nsteps=$STEPS \
    -kfold=$K \
    -method=$METHOD \
    -folder=$(dirname $OUTPUT_PREFIX) \
    -prefix=$(basename $OUTPUT_PREFIX) \
    -sample=$SAMPLES \
    -labels=$LABELS
