# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:23:14 2016

@author: liulei
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
 
# 加载数据
# 备用地址: http://cdn.powerxing.com/files/lr-binary.csv
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# 重命名'rank'列，因为dataframe中有个方法名也为'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]

# 虚拟变量
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
data['intercept'] = 1.0


# 预测
#  更改 ： 将dataset分为训练集和测试集， 训练集拟合， 测试集预测
import sklearn.cross_validation

train_cols = data.columns[1:]
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(data[train_cols], data['admit'],test_size=0.3, random_state=5)

# 执行逻辑回归 
logit = sm.Logit(Y_train, X_train)
# 拟合模型
result = logit.fit()
pred_test=result.predict(X_test)   # 1行 n列

total=0
# 把pred_test 和  Y_test 对应起来比较
pred_test=pd.Series(pred_test)

for value in pred_test:
    if value > 0.5:   
        value=1
    else:
        value=0

#predict_cols = combos.columns[1:]
# 
## 预测集也要添加intercept变量
#combos['intercept'] = 1.0
# 
## 进行预测，并将预测评分存入 predict 列中
#combos['predict'] = result.predict(combos[predict_cols])
 
# 预测完成后，predict 的值是介于 [0, 1] 间的概率值
# 我们可以根据需要，提取预测结果
# 例如，假定 predict > 0.5，则表示会被录取
# 在这边我们检验一下上述选取结果的精确度
#total = 0
#hit = 0
#for value in combos.values:
#  # 预测分数 predict, 是数据中的最后一列
#  predict = value[-1]
#  # 实际录取结果
#  admit = int(value[0])
# 
#  # 假定预测概率大于0.5则表示预测被录取
#  if predict > 0.5:
#    total += 1
#    # 表示预测命中
#    if admit == 1:
#      hit += 1
# 
## 输出结果
#print 'Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)