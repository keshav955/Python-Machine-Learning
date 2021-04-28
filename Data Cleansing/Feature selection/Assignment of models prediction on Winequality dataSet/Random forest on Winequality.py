# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:47:50 2020

@author: keshav
"""


# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')

dataset.isna().sum()
dataset.dtypes

X = dataset.iloc[:,0:11]    
y = dataset.iloc[:,11]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

mse =np.mean((y_pred - y_test)**2)
