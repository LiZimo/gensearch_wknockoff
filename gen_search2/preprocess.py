#!/usr/bin/env python

'''
Search for patterns using the learn/predict functions provided by scikit-learn library.

There are 3 files involved in every run:
1) A csv input file, which has samples as rows, and variables as columns.
2) A csv labels files, has the same samples as in the input file and,
   the class (0/1) as a row.
3) Params file, which determines the run parameters.
'''

from __future__ import division
import argparse
import ConfigParser
import sys
import csv
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import itertools as it
import math
import pickle
import os

#Read in feature and label files
def read_csv(csv_file):
    data = pd.read_csv(csv_file, sep=',', header=0, index_col=0)
    print("Read file : %s" % csv_file)
    return data

#Read in parameter file
def read_params(params_file):
    cfg = ConfigParser.ConfigParser()
    cfg.read(params_file)
    try:
        n = cfg.getint("params", "n")
        nsteps = cfg.getint("params", "nsteps")
        k = cfg.getint("params", "k")
        method = cfg.getint("params", "method")
        if method < 1 or method > 2:
            print("ERROR : Method must be between 1 and 2\n\tmethod 1: test and train on same data\n\tmethod 2: k-fold cross validation")
            sys.exit(1)
        print("Read parameters : n = %d, nsteps = %d, k = %d, method = %d" % (n, nsteps, k, method))
    except:
        print("ERROR : Could not read parameter file")
        sys.exit(1)
    return n, nsteps, k, method

#Check that classes on features and labels match
def check_consistency(inputs, labels):
    if (inputs.index.all() == labels.index.all()):
        print("Consistency check : Indices match")
    else:
        print("ERROR : Indices do not match")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Filename of samples file in csv. This argument is required')

    parser.add_argument('-l', '--labels', required=True,
                        help='Filename of the labels file in csv. This argument is required')

    parser.add_argument('-p', '--params', required=False,
                        help='Filename of the sets of combinations')

    parser.add_argument('-x', '--splitnum', required=True,
                        help='Number of chunks the problem space is split into')

    parser.add_argument('-f', '--prefix', required=True,
                        help='Prefix in the form <directory/file_name_prefix>')

    parser.add_argument('-v', '--verbose', action="store_true", help='Turn on verbose output')

    parser.add_argument('-n', '--n', required=False,
                        help='The index number of slice ?')

    parser.add_argument('-s', '--nsteps', required=False,
                        help='The number of steps ?')

    parser.add_argument('-k', '--kval', required=False,
                        help='The number of steps ?')

    parser.add_argument('-m', '--method', required=False,
                        help='Choose method 1/2 ?')


    print '\n{s:{c}^{n}}\n'.format(s='Sanity Checks',n=106,c='-')
    args = parser.parse_args()
    inputs = read_csv (args.input)
    labels = read_csv (args.labels)
    param  = args.params
    n      = int(args.n)
    nsteps = int(args.nsteps)
    k      = int(args.kval)
    method = int(args.method)
    prefix = args.prefix
    splitnum = int(args.splitnum)

    print("Arg parameters : n = %d, nsteps = %d, k = %d, method = %d" % (n, nsteps, k, method))

    #Exit immediately if consistency checks do not pass
    check_consistency (inputs, labels)

    #List of possible k-combinations
    sets = list(it.combinations(range(inputs.shape[1]), n))
    numSets = len(sets)
    print "Splitting ", numSets," combinations to ", splitnum, " chunks"

    if not os.path.exists(os.path.dirname(prefix)):
        print "Creating directory :", os.path.dirname(prefix)
        os.makedirs(os.path.dirname(prefix))

    chunk_size = int(numSets / splitnum)
    for i in range(0, numSets, chunk_size):
        filename = prefix+'_'+str(int(i/chunk_size))+".split"
        print "filename : ", filename , "indices : ", i, i+chunk_size
        pickle.dump(list(sets[i:i+chunk_size]), open(filename, 'wb'))



