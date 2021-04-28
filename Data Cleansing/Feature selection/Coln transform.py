# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:52:27 2020

@author: keshav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
labelencoder = LabelEncoder()

"""
name: a name for the column transformer, which will make setting of parameters and 
searching of the transformer easy.transformer: here we’re supposed to provide an estimator.
 We can also just “passthrough” or “drop” if we want.But since we’re encoding the data in 
 this example, we’ll use the OneHotEncoder here. Remember that the estimator you use here 
 needs to support fit and transform. column(s): the list of columns which you want 
 to be transformed. In this case, we’ll only transform the first column.
"""