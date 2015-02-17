# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:02:38 2015

@author: Asus
"""
import operator
import csv

def cutoff(pair_list, percentage, output_file): ## pair_list is a list where each entry is: [percentage, feature1, feature2]
    f= open(pair_list, 'r+')
    reader = csv.reader(f)
    listform = list(reader)

    cut_list = []
    for row in listform:
        if float(row[0])>=float(percentage):
            cut_list.append(row)
    
    dict_count = {}
    for entry in cut_list:
        for i in range(1, 3):
            feature = entry[i]
            try: dict_count[feature] += 1
            except: dict_count[feature] = 0
    
    outputlist = sorted(dict_count.items(), key = operator.itemgetter(1), reverse = True)
    
    out = open(output_file, 'w')
    
    for entry in outputlist:
        out.write(str(entry))
        out.write('\n')
    
    out.close()
    return outputlist

x = cutoff("output/IrisLogisW100noise.csv", 1.0, "output/IrisLogis100cutoff.csv")