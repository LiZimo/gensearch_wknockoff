# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 14:14:03 2015

@author: Asus
"""

#==============================================================================
# 
# How to use:
#     
# The function named "knockoff(X)" will provide a knockoff matrix for X, 
# where X is a matrix in the form of a list of lists.
#==============================================================================

import math
import numpy as np;
from scipy import linalg;
from numpy import array;
from numpy import matrix;
from numpy import transpose;
from numpy.linalg import inv;
from numpy.linalg import norm;
import matplotlib.pylab as plt;
from random import randint;

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'




def GramSchmidt(X): ## given array X, return matrix (as a list) of orthonormal basis of columns of X.  Same dimensions as X.
    rowsA,colsA=X.shape
    #Initialize the Y vector
    Y=np.zeros(rowsA)
    #Initialize Q, the array of vectors which are the vectors orthogonal to the vectors in X
    Q=np.zeros([rowsA,colsA])
    #Step 1
    #np.inner is the inner product of vectors for 1-D arrays
    #X[:,0] gives us the first column vector
    productX1 = 1.0 / np.sqrt(np.inner(X[:,0],X[:,0]))
    #Steps 2 through 5
    Q[:,0]=productX1*X[:,0]
    for j in range(1,colsA):
        #Step 3
        Y=X[:,j]
        for i in range(0,j):
            Y=Y-np.inner(X[:,j],Q[:,i])*Q[:,i]
        #Step 4
        productYj=1.0 / np.sqrt(np.inner(Y,Y))
        Q[:,j]= productYj*Y
    return Q.tolist()

def diagonalize(vector): ## vector is a list or a vector of floats
    length=len(vector)
    mat= []
    for i in range(length):
        copy = [0]*len(vector)
        copy[i]=vector[i]
        mat.append(copy)
        
    return matrix(mat)
 

def inner_iszero(vec1, vec2):
    if int(np.inner(vec1, vec2)) == 0:
        return True
    else:
        return False

def Orthogonal_to(X): ## assumes numrows >= numcolumns
    column_list = transpose(matrix(X)).tolist()
    for i in range(len(X)):
        vec = [0]*len(X)
        vec[i]=1
        column_list.append(vec)
    gs = GramSchmidt(transpose(array(column_list))) ## columns are the orthogonal vectors
    


    gs = transpose(matrix(gs)).tolist() ## now rows are orthogonal vectors
    prelim_output = []
    
    for index in range(len(gs)-1, len(X[0])-1, -1):
        prelim_output.append(gs[index])
     
    final_output = prelim_output[:]

    for item in prelim_output:
        prelim_output.remove(item)
        for other in prelim_output:
            if np.allclose(array(item), array(other)):
                final_output.remove(item)
                break
            elif np.allclose((-1)*array(item), array(other)):
                final_output.remove(item)
                break
                
    #return final_output[1:len(X[0])+1]
    return final_output[len(final_output)-1:len(final_output)-len(X[0])-1:-1]
    
def normalize_columns(X):
    output = []
    column_list = transpose(matrix(X)).tolist()
    for column in column_list:
        new_column = array(column)/math.sqrt(np.inner(column, column))
#        column_mean = np.mean(column)
#        column_std = np.std(column)
#        new_column = []
#        for element in column:
#            new_elm = (element - column_mean)/column_std
#            new_column.append(new_elm)
        output.append(new_column)
    output = transpose(matrix(output)).tolist()
    return output

def perform_knockoff(M, s): ## s is a vector or 1-D array, X is the data matrix, input as a list of lists
    diagonal = [s]*len(M[0]);
    
    X = normalize_columns(M)
    num_columns = len(X[0]);
    gram = transpose(matrix(X))*matrix(X);  
#    print gram, "gram"
#    raw_input()
    gram_inverse = inv(gram);
    
    I = [1]*num_columns;
    I = diagonalize(I);
    diag_s = diagonalize(diagonal);
    

    U = transpose(matrix(Orthogonal_to(X)))
    
    A = 2*diag_s - diag_s*gram_inverse*diag_s;
    #A = array(A)
    
    C = matrix(linalg.cholesky(A))
    
    output = matrix(X)*(I - gram_inverse*diag_s) + U*C
#    print color.BOLD + "\n===============the original gram matrix ===============\n\n" + color.END, gram
#    print color.BOLD + "\n===============transpose(knockoff) x knockoff; should be same as gram matrix ===============\n\n" + color.END, transpose(output)*output 
#    print color.BOLD +"\n===============transpose(knockoff) x original matrix; should differ only on diagonal by s===============\n\n" + color.END, transpose(output)*matrix(X)
#    print "got a knockoff"
#    raw_input()
    return output.tolist();

 

#X = []
#for j in range(6):
#    first = float(randint(0, 15))
#    second = float(randint(0,15))
#    third = float(randint(0,15))
#
#
#    new_row = [first, second, third]
#    X.append(new_row)
#X_tilde = perform_knockoff(X, 0.1)
#
#print X_tilde
#raw_input()
#print normalize_columns(X_tilde)