# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:55:34 2020

@author: keshav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]

# taking Care of nan Values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=impute.transform(X.iloc[:, 1:3])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
b = ct.fit_transform(X)