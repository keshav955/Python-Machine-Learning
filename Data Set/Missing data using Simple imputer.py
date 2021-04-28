# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:28:09 2020

@author: keshav
"""
#Handling The Null Values

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

#importing the Dataset

dataset = pd.read_csv('Data.csv')

dataset.isna().sum()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,3]

# Divide the dataset into independent and dependent variables 

from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=impute.transform(X.iloc[:, 1:3])