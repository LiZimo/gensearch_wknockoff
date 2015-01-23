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
#==============================================================================
# 
#
#X = []
#for j in range(45):
#    first = float(randint(0, 15))
#    second = float(randint(0,15))
#    third = float(randint(0,15))
#    fourth = float(randint(0,15))
#    fifth = float(randint(0,15))
#
#    new_row = [first, second, third, fourth, fifth]
#    X.append(new_row)
#
#
##K = [[1.,3.,1.],[3.,2.,6.],[5.,1.,8.],[1.,3.,9.],[1.,1.,2.],[4.,1.,1.]]
#X = [[ 3.16885162 , 2.88860441],[ 3.09953752 , 2.95355167],[ 2.99976541 , 2.99976541],[ 3.28898209 , 3.03971766],[ 3.19367609 , 3.21721046],[ 3.49404657 , 3.49404657],[ 3.0658405  , 3.10677575],[ 2.64622355 , 2.03278481],[ 2.8409916  , 2.81650375],[ 3.45407205 , 3.45407205],[ 3.3136573  , 3.3136573 ],[ 3.89821831 , 3.89821831],[ 2.79831436 , 2.76286532],[ 3.28764889 , 2.99666491],[ 3.08083138 , 2.9699706 ],[ 3.04060904 , 3.21635719],[ 2.72396157 , 2.54882779],[ 3.68131148 , 3.68131148],[ 3.37292641 , 3.49006333],[ 2.33296074 , 2.18525055],[-0.48271206 ,-1.06220892],[ 3.55606819 , 3.3615511 ],[ 3.70836959 , 3.70836959],[ 2.95446596 , 2.86503476],[ 3.45335107 , 3.45161081],[ 2.57602737 , 2.81348551],[ 3.09344039 , 2.81872963],[ 3.18634522 , 2.95821007],[ 3.56216758 , 3.64076143],[ 2.89050114 , 2.79373191],[ 3.27934473 , 3.25932539],[ 3.06642134 , 2.84412463],[ 2.88575959 , 2.7472807 ],[ 3.30283328 , 3.17150411],[ 3.30750982 , 2.91009709],[ 2.81781214 , 2.81781214],[ 3.01414246 , 2.96104422],[ 3.2756409  , 3.21914079]]
#X = normalize_columns(X)
#X_tilde = perform_knockoff(X, 0.003)
#print X_tilde

#gram_inverse = inv(transpose(matrix(X))*matrix(X))
##
#diag = matrix([[.1,0.0],[0.0, .1]])
##
#A = 2*diag - diag*gram_inverse*diag
#C = linalg.cholesky(array(A))
##
##print gram_inverse
#print A, "A"
#print gram_inverse, "gram inverse"
##print C
