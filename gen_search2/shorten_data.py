# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 22:41:19 2015

@author: Asus


"""
import csv
import numpy as np
from numpy import matrix
from numpy import transpose

                                                ## csv_sample_file is the filepath as a string
def shorten_samples(csv_sample_file, scaling_factor, output): ## how much is what factor you reduce data by.  Must be greater than or equal to 1
    f= open(csv_sample_file, 'r+')
    reader = csv.reader(f)
    listform = list(reader)
    
    new_end_index = int(len(listform[0])/scaling_factor)
    

    for i in range(len(listform)):
        listform[i]=listform[i][0:new_end_index]
    print len(listform)
    print len(listform[0])
    
    output_file_name = output 
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


shorten_samples('data/golub_samples.csv', 100, 'data/golub_samples_gimped.csv');