# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:09:36 2020

@author: keshav
"""


from pydoc import help
from scipy.stats.stats import pearsonr

import pandas as pd 
import seaborn as sns
df = pd.read_csv('cars.csv')
cor=pearsonr(df.iloc[:,2],df.iloc[:,4])
print(cor)


data =  pd.read_csv('cars.csv')

corr = data.corr()
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