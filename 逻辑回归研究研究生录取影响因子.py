# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 19:47:14 2016

@author: liulei
"""

'''

逻辑回归研究不同的因素对研究生录取的影响

'''

import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression 
import pylab as pl
import numpy as np


df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

#重命名
df.columns = ["admit", "gre", "gpa", "prestige"]
#print pd.crosstab(df['prestige'],df['admit'] , rownames=['prestige'])

#df.hist()
#pl.show()

dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')

#print dummy_ranks.head()

clos_to_keep = ["admit", "gre", "gpa"]
data = df[clos_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])

#print data.head()

import sklearn.cross_validation

train_cols = data.columns[1:]
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(data[train_cols], data['admit'],test_size=0.3, random_state=5)

logit = LogisticRegression()
logit.fit(X_train, Y_train)

pred_test=logit.predict(X_test)   # 1行 n列

p = np.mean(pred_test == Y_test)
print p

from sklearn.metrics import precision_recall_curve  

answer = logit.predict_proba(X_test)[:,1]  
#准确率与召回率
precision, recall, thresholds = precision_recall_curve(Y_test, answer)  
    
