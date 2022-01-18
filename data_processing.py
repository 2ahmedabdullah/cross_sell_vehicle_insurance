#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 21:26:17 2022

@author: abdul
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from utils import *
import pickle


train = pd.read_csv('train.csv')

y = train['Response']
x = train.drop(['id','Response', 'Region_Code'], axis=1)



train['Response'].value_counts().plot.bar(figsize=(5, 5), rot=0)
plt.suptitle('Class wise distribution', fontsize=18)
plt.xlabel('class', fontsize=10)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'class_wise_distribution.png')
plt.show()

print(round(sum(train['Response'])/len(train),2))

list(x)

##NAN in each column
nans = train.isnull().sum(axis = 0)

#PLOTTING CARDINALITY OF EACH FEATURE
unique_values = train.nunique().sort_values(ascending=False)

unique_values.plot.bar(figsize=(12, 6), rot=1)
plt.suptitle('Cardinality of Each Feature', fontsize=18)
plt.xlabel('features', fontsize=10)
plt.ylabel('counts', fontsize=10)
plt.savefig(plot_path+'cardinality.png')    
plt.show()


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#correlation
corr_coeff = 0.85
corr_matrix = X_train.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_feature2 = [column for column in upper.columns if any(upper[column] > corr_coeff)]


samples = pd.concat([X_train, y_train], axis=1)
zeros = samples.loc[samples['Response'] == 0]
ones = samples.loc[samples['Response'] == 1]


x_0 = zeros.sample(frac=0.3, replace=False, random_state=1)
x_train = pd.concat([x_0, ones], axis=0)
y_train = x_train['Response']
x_train = x_train.drop(['Response'], axis=1)


num_var = ['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

cat_var = ['Gender', 'Vehicle_Age', 'Driving_License', 'Previously_Insured', 'Vehicle_Damage']

import seaborn as sns


train_num, train_cat = num_cat(x_train, num_var, cat_var)
test_num, test_cat = num_cat(X_test, num_var, cat_var)

fig = plt.hist(train_num['Age'], bins = 100,  by=y_train, figsize = (15, 5))

fig, ax = plt.subplots(figsize = (15, 5))
train_num['Age'].hist(bins = 100,  by=y_train, ax=ax)
plt.suptitle('Age Distribution')
fig.savefig(plot_path+'Age.png')

fig, ax = plt.subplots(figsize = (15, 5))
train_num['Annual_Premium'].hist(bins = 100, by=y_train, ax=ax)
plt.suptitle('Annual_Premium Distribution')
fig.savefig(plot_path+'prem.png')

fig, ax = plt.subplots(figsize = (15, 5))
train_num['Policy_Sales_Channel'].hist(bins = 10,  by=y_train, ax=ax)
plt.suptitle('Policy_Sales_Channel Distribution')
fig.savefig(plot_path+'chanel.png')

fig, ax = plt.subplots(figsize = (15, 5))
train_num['Vintage'].hist(bins = 1000,  by=y_train, ax=ax)
plt.suptitle('Vintage Distribution')
fig.savefig(plot_path+'vintage.png')


ax = train.groupby(['Gender', 'Response']).size().unstack().plot.bar()
fig = ax.get_figure() 
fig.savefig(plot_path+'gender.png')


ax = train.groupby(['Vehicle_Age', 'Response']).size().unstack().plot.bar()
fig = ax.get_figure() 
fig.savefig(plot_path+'Vehicle_Age.png')


ax = train.groupby(['Driving_License', 'Response']).size().unstack().plot.bar()
fig = ax.get_figure() 
fig.savefig(plot_path+'Driving_License.png')


ax = train.groupby(['Previously_Insured', 'Response']).size().unstack().plot.bar()
fig = ax.get_figure() 
fig.savefig(plot_path+'Previously_Insured.png')


ax = train.groupby(['Vehicle_Damage', 'Response']).size().unstack().plot.bar()
fig = ax.get_figure() 
fig.savefig(plot_path+'Vehicle_Damage.png')



X_train_transform = transformation(train_num, train_cat) 
X_test_transform = transformation(test_num, test_cat)

X_train_transform = X_train_transform.astype('float')
X_test_transform = X_test_transform.astype('float')


#SAVE PICKLE
pickle.dump(X_train_transform, open('x_train.pkl', 'wb'))
pickle.dump(X_test_transform, open('x_test.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))


