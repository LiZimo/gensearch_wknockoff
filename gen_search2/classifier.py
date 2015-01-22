#!/usr/bin/env python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#clf = SVC(kernel="linear", C=1)
clf = LogisticRegression(penalty='l1', dual=False, tol=0.01, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
