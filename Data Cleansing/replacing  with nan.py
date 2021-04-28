# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:39:56 2020

@author: keshav
"""

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

def check(x):
   if str(x).isnumeric():
       return x
   
   else:
       return np.NaN
       
    
dataset['Age']=dataset['Age'].apply(check)    