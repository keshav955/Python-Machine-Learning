# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:43:01 2020

@author: keshav
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cars.csv')

dataset.isna().sum()

dataset.dtypes

def check(x):
    try :    
        
        float(x) 
        res = True
        return x
    except :         
        res = False
    #    print("Not a float Value")
        return np.NaN
    
dataset['horsepower']=dataset['horsepower'].apply(check)   

dataset.dtypes

X = dataset.iloc[:,1:8]    
y = dataset.iloc[:,0:1]


# taking Care of nan Values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(X.loc[:,['horsepower']])
#impute.fit(X.iloc[:,2])
X.loc[:,['horsepower']] =impute.transform(X.loc[:,['horsepower']])
#X.iloc[:, 2]=impute.transform(X.iloc[:,2])

#X.dtypes

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

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test)



#finding The Mean Square Error
mse =np.mean((y_pred - y_test.iloc[:,0])**2)

