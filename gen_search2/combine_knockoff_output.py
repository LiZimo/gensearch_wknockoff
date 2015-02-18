# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:42:39 2015

@author: Asus
"""
from collections import Counter;
import csv;
import operator;
import os;

def merge_dicts(dict1, dict2): # two dictionaries
    A = Counter(dict1)    
    B = Counter(dict2)
    return A + B;
    
    
def knock_f_to_dict(filename): # string of filename
    f= open(filename, 'r+')
    reader = csv.reader(f)
    listform = list(reader)

    my_dict = {}
    for row in listform:
        my_dict[row[1]] = row[0]
    
    f.close();
    
    return my_dict;
    
def combine_file_w_dict(file1, my_dict): ## file1 and file2 are strings of the filenames
   dict1 = knock_f_to_dict(file1);
   merged = merge_dicts(dict1, my_dict);
   
   results = sorted(merged.items(), key=operator.itemgetter(1), reverse=True)
   
   return dict(results);
   

def ko_merge_directory(dir_name, outfile):
    true_dict = {}    
    
    for filename in os.listdir(dir_name):
        true_dict = combine_file_w_dict(filename, true_dict)
    
    results = sorted(true_dict.items(), key=operator.itemgetter(1), reverse=True)
    
    fp = open(outfile, 'wb')
    w = csv.writer(fp)
    # No need for fancy notices
    #w.writerow(["accuracy"] + ["feature_"+str(i) for i in range(n)] + ["feature_"+str(i)+"_index" for i in range(n)])
    for x in results[:1000]:
        w.writerow(x)
    fp.close()