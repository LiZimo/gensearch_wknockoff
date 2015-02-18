#!/bin/bash


SAMPLES=data/miRNA_samples.csv
LABELS=data/miRNA_labels.csv
N=2
STEPS=20
K=5
METHOD=2
SPLITNUM=100
# This must be set as a directory/prefix
OUTPUT_PREFIX=output/miRNA

# This section initializes the modules system and loads the appropriate modules
source /cvmfs/oasis.opensciencegrid.org/osg/modules/lmod/5.6.2/init/bash
module load python/2.7
module load libgfortran
module load atlas
module load lapack
module load all-pkgs


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
