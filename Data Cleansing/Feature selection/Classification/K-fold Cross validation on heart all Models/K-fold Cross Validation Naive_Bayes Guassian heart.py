# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:45:27 2020

@author: keshav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('heart.csv')

convert={'target':str}
dataset =  dataset.astype(convert)

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

dataset.dtypes


from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=42,shuffle=True)

total_acc = [] 

for train_index, test_index in kf.split(X):
   # print(train_index)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
       
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting a new result
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(cm)
    
    accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] +cm[1,0] + cm[1,1])*100
    
    total_acc.append(accuracy)
    
    print(accuracy)
    
average_acc = sum(total_acc) / 5
print(average_acc)   


