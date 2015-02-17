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

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import argparse
import ConfigParser
import sys
import csv
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import itertools as it
import math
from matplotlib import pyplot as plt
import sklearn
from sklearn import cross_validation, metrics
import classifier
import knockoff
import operator
import pickle



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

#calculate measures from k-fold cross validation
def kfold(X, y, clf, k):
    nclasses = len(np.unique(y))
    acc = 0
    kf = cross_validation.KFold(len(y), n_folds=k, indices=True, shuffle=True, random_state=np.random.randint(100))
    while (any([len(np.unique(y[test]))<nclasses or len(np.unique(y[train]))<nclasses for test, train in kf])):
        kf = cross_validation.KFold(len(y), n_folds=k, indices=True, shuffle=True, random_state=np.random.randint(100))
    for train, test in kf:
        y_pred = clf.fit(X[train], y[train]).predict(X[test])
        acc += metrics.accuracy_score(y[test], y_pred)
    return acc/k

#Run k-fold cross validation multiple times, return mean accuracy
def kfold_multi_run(X, y, clf, k, nsteps):
    acc = 0
    for i in range(nsteps):
        acc += kfold(X, y, clf, k)
    return acc/nsteps

#Test and train on the same data
def run(X, y, clf):
    y_pred = clf.fit(X, y).predict(X)
    acc = metrics.accuracy_score(y, y_pred)
    return acc

# get the coefficient matrix back from the classifier
def return_coeff(X, y,clf):
    coefficients = clf.fit(X, y).coef_
    ##print coefficients
    ##raw_input()
    return coefficients
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Filename of samples file in csv. This argument is required')

    parser.add_argument('-l', '--labels', required=True,
                        help='Filename of the labels file in csv. This argument is required')

    parser.add_argument('-p', '--params', required=False,
                        help='Filename of the sets of combinations')

    parser.add_argument('-o', '--output', required=True,
                        help='Filename of output file to generate. Required')

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
    outfile= args.output
    print("Arg parameters : n = %d, nsteps = %d, k = %d, method = %d" % (n, nsteps, k, method))

    #Exit immediately if consistency checks do not pass
    check_consistency (inputs, labels)

    #List of possible k-combinations
    sets = list(it.combinations(range(inputs.shape[1]), int(n)))
    #sets = pickle.load(open(param,"rb"))
    numSets = len(sets)
    print '\n{s:{c}^{n}}\n'.format(s='Running for %d Feature Combinations' % numSets,n=106,c='-')

    #Run for all possible n-combinations of feature sets
    y = np.array(labels[labels.columns[0]])
    sys.stdout.write('0%')
    sys.stdout.flush()
    percentDone = 0
    results = []
    mydict = {}
    for i in range(numSets):
    
        
        X = inputs[list(sets[i])].values


        if method == 1:
            acc = run(X, y, classifier.clf)
            coeffs = return_coeff(X,y,classifier.clf)          
        elif method == 2:
            acc = kfold_multi_run(X, y, classifier.clf, k, nsteps)
            coeffs = return_coeff(X,y,classifier.clf).tolist()
        if method == 3:
            X_reg = inputs[list(sets[i])].values
            got_knockoff = False
            s = 1
            while got_knockoff == False:
            
                try: 
                    X_tilde = knockoff.perform_knockoff(list(X_reg),s)
                    got_knockoff = True
                    break
                except: 
                    s = s*(0.75)
                    continue
            
            
            X = np.concatenate(((knockoff.normalize_columns(X_reg)),knockoff.normalize_columns(X_tilde)), axis = 1)

            our_clf = classifier.clf
            feat1_done = False
            feat2_done = False
            regul = 11.0
            
            step = 2.0
            last_change = 0
            while feat1_done == False:
                coeffs = our_clf.fit(X,y).coef_.tolist()[0]
#                coeffs_round = list(round(x, ) for x in coeffs)
#                coeffs = coeffs_round
                

                if coeffs[0]==0.0 and coeffs[2]==0.00:
                    if last_change == -1:
                        step = step/2
                    regul = regul + step
                    last_change = 1
                    print "increased regularization 0"
#                    raw_input()
                elif coeffs[0]!=0.0 and coeffs[2]==0.00:
                    try: mydict[inputs[list(sets[i])].columns[0]] += 1
                    except: mydict[inputs[list(sets[i])].columns[0]] = 1
                    feat1_done = True
                    print "done 0"
#                    print "feat1 came before knockoff"
#                    raw_input()
                elif coeffs[0]==0.0 and coeffs[2]!=0.00:
                    feat1_done = True
                    print "done 0"
#                    print "feat 1 came after knockoff"
#                    raw_input()
                elif coeffs[0]!=0.0 and coeffs[2]!=0.00:
                    if last_change == 1:
                        step = step/2
                    regul = regul - step
                    last_change = -1
                    print "decreased regularization 0"
#                    raw_input()
                our_clf.set_params(C = regul)
            
            regul = 11.0
            step = 2.0
            last_change = 0
            while feat2_done == False:
                coeffs = our_clf.fit(X,y).coef_.tolist()[0]
#                print "got new coeffs"
#                print coeffs
#                raw_input()
                if coeffs[1]==0.0 and coeffs[3]==0.00:
                    if last_change == -1:
                        step = step/2
                    regul = regul + step
                    last_change = 1
                    print "increased regularization 1"
#                    raw_input()
                elif coeffs[1]!=0.0 and coeffs[3]==0.00:
                    try: mydict[inputs[list(sets[i])].columns[1]] = mydict[inputs[list(sets[i])].columns[1]] + 1
                    except: mydict[inputs[list(sets[i])].columns[1]] = 1
                    feat2_done = True
                    print "done 1"
#                    print "feat2 came before knockoff"
#                    raw_input()
                elif coeffs[1]==0.0 and coeffs[3]!=0.00:
                    feat2_done = True
                    print "done 1"
#                    print "feat 2 came after knockoff"
#                    raw_input()
                elif coeffs[1]!=0.0 and coeffs[3]!=0.00:
                    if last_change == 1:
                        step = step/2
                    regul = regul - step
                    last_change = -1
                    print "decreased regularization 1"
#                    raw_input()
                our_clf.set_params(C = regul)
            
        if method!=3:
            results.append([round(acc,2)] + [x for x in inputs[list(sets[i])].columns]+ coeffs)
        if i/numSets >= percentDone:
            sys.stdout.write('...'+ str(percentDone*100))
            sys.stdout.flush()
            percentDone += 0.01
    sys.stdout.write('100%\n')
    sys.stdout.flush()

    #Rank results by highest accuracy:
    if method!=3:
        results.sort(key=lambda x: x[0], reverse=True)

    if method == 3:
        results = sorted(mydict.items(), key=operator.itemgetter(1), reverse=True)
#        percentages = []
#        for entry in results:
#            new_entry = list(entry)
#            new_entry[1]=float(new_entry[1]/((inputs.shape[1])-1))
#            percentages.append(new_entry)
#        results = percentages
    fp = open(outfile, 'wb')
    w = csv.writer(fp)
    print '\n{s:{c}^{n}}\n'.format(s='Writing Results',n=106,c='-')
    # No need for fancy notices
    #w.writerow(["accuracy"] + ["feature_"+str(i) for i in range(n)] + ["feature_"+str(i)+"_index" for i in range(n)])
    for x in results[:1000]:
        w.writerow(x)
    fp.close()
    print("Done!")


