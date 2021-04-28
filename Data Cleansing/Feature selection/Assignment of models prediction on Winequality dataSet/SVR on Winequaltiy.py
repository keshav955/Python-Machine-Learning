# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 23:22:30 2020

@author: keshav
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')

dataset.isna().sum()
dataset.dtypes

X = dataset.iloc[:,0:11]    
y = dataset.iloc[:,11:12]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

#finding The Mean Square Error
mse =np.mean((y_pred - y_test.iloc[:,0])**2)

