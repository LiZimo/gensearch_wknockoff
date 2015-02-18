#!/bin/bash


SAMPLES=data/miRNA_samples.csv
LABELS=data/miRNA_labels.csv
N=2
STEPS=20
K=5
METHOD=2
SPLITNUM=1000
OUTPUT_PREFIX=output/miRNA
#./preprocess.py -i  -l  -n 2 -s 20 -k 5 -m 2 -x 1000 -f output/miRNA

if [ ! -f "python_wrapper.sh" ]
then
    echo "ERROR: python_wrapper.sh  is missing"
    exit 0
fi

source python_wrapper.sh

which swift | grep "Swift 0.95 RC6"
if [[ "$?" != "0" ]]
then
    echo "ERROR: Must run script with Swift 0.95 RC6 or higher"
fi

rm -rf $OUTPUT_PREFIX* &> /dev/null
python preprocess.py -i $SAMPLES -l $LABELS -n $N -s $STEPS -k $K -m $METHOD -x $SPLITNUM -f $OUTPUT_PREFIX

swift gen_search.swift