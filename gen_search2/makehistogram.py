# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 18:59:49 2015

@author: Asus
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

def make_histo(knock_f ): ## file of knockoff regression data
    
    f= open(knock_f, 'r+')
    reader = csv.reader(f)
    listform = list(reader)
    
    
    val_list = []
    for row in listform:
        val_list.append(int(row[1]))

    #np.histogram(val_list, int(max(val_list)))
    
    plt.hist(val_list, 2*int((max(val_list))))
    plt.title("All Uniform run 2 (red) and Uniform w/ real (blue) ")
    plt.xlabel("# times entered before knockoff")
    plt.ylabel("Frequency")
    plt.show()
    
make_histo("random_runs/output/55uniform1to5a.csv")