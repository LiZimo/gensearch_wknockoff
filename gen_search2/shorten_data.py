# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:41:19 2015

@author: Asus


"""
import csv
import numpy as np
import random
from numpy import matrix
from numpy import transpose

                                                ## csv_sample_file is the filepath as a string
def shorten_samples(csv_sample_file, scaling_factor, out, desired_columns, num_noisy_columns): ## how much is what factor you reduce data by.  Must be greater than or equal to 1
    f= open(csv_sample_file, 'r+')
    reader = csv.reader(f)
    listform = list(reader)
    print len(listform)
    print len(listform[0])
    
    new_end_index = int(len(listform[0])/scaling_factor)
    

    for i in range(len(listform)):
        listform[i]=listform[i][0:new_end_index]

    
    if len(desired_columns)!=0 or num_noisy_columns!=0:
        desired = get_target_columns(csv_sample_file, desired_columns, num_noisy_columns)    
        listform = np.concatenate((listform, desired), axis = 1)
    
    output_file_name = out
    output = open(output_file_name, 'w')
    for row in listform:
        
        for i in range(len(row)):
            entry = row[i] + ','
            if i == len(row)-1:
                entry = row[i]
            output.write(entry)
            
        output.write('\n')
    output.close()
    f.close()
    
def get_target_columns(csv_sample_file, column_names, add_noisy_columns): ## column names is al ist of desired features
    f= open(csv_sample_file, 'r+')
    reader = csv.reader(f)
    listform = list(reader)
    
    column_indices = []
    
    for i in range(len(listform[0])):
        for name in column_names:
            if str(listform[0][i]) == name:
                column_indices.append(i)
    
    output = []

    for j in column_indices:
        output.append(np.array(listform)[:,j].tolist())
        
    for i in range(add_noisy_columns):
        noisy_col = []
        for j in range(len(listform)):
            noisy_col.append(random.uniform(0,5))
        output.append(noisy_col) 
    f.close
    return transpose(output)

names = ['sepal length','sepal width','petal length','petal width']
#newlist = get_target_columns('data/iris_samples.csv', names)

shorten_samples('data/iris_samples.csv', 100, 'random_runs/data/205uniform0to5.csv', [] , 205);