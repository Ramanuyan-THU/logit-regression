# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:36:17 2016

@author: liulei
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


'''
一维 实例
'''

dat_1d = {"x"    :[2, 2, 1, 0.25, -0.4, -1, -2, -2, 1], 
          "label":[0, 0, 0,    0,    1,  1,  1,  1, 1]}
          
          
df_1d = pd.DataFrame.from_dict(dat_1d)

#根据 y 值进行分类
df_1d["label_fact"] = pd.Categorical.from_array([str(x) for x in df_1d.label])

#plt.scatter(df_1d.loc[:,'x'],df_1d.loc[:,'label'])
#plt.show()

def sigmoid(x):
    # vector in, vector out
    return 1./(1. + np.exp(-x))

def cost(label_arr, model_arr):
    # inputs are nb.array([])
    # sum over training set
    c = -(1. - label_arr)*np.log(1. - model_arr) - label_arr*np.log(model_arr)
    return np.sum(c)/len(c)
    
def model_1d(model_vec, df):
    # Calculate model output
    z = model_vec[0] + model_vec[1] * df.x
    return sigmoid(z)
    
def model_cost(model_vec, df, model):
    # model_vec is python vector
    # df is data frame with column x and column label
    # model is a functionto calculate model output
    return cost(df.label, model(model_vec, df))

from scipy.optimize import minimize as mini

res_1d = mini(model_cost, x0=[0.1,0.1], args=(df_1d, model_1d)) 
print (res_1d)


# 预测
#df_pred_1d = pd.DataFrame.from_dict({"x": np.linspace(-3,3,99)})
#df_pred_1d["y"] = model_1d(res_1d.x, df_pred_1d)
#df_pred_1d.head()


#print classifier.decision_function()