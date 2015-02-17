#!/bin/bash

source  /cvmfs/oasis.opensciencegrid.org/osg/modules/lmod/lmod/init/bash
module load python/2.7
module load libgfortran
module load atlas
module load lapack
module load all-pkgs

ls -thor
echo "Launching : python $*"
#python $*