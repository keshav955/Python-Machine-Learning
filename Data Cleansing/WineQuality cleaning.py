# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 14:46:40 2020

@author: keshav
"""


import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('winequality-red.csv')

dataset.isna().sum()


from pydoc import help
from scipy.stats.stats import pearsonr

cor=pearsonr(dataset.iloc[:,0],dataset.iloc[:,11])

print(cor)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]

# taking Care of nan Values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=impute.transform(X.iloc[:, 1:3])


corr = dataset.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

