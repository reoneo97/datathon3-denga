# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:53:55 2020

@author: reone
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout
df = pd.read_csv("Processed_Data.csv",index_col = 0)
train = df[:"2019"]
test = df["2019":]
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
train = sc.fit_transform(train)

max_lookback = 26
pred_interval = 8 
X_train =[]
y_train = [] 
for i in range(len(train)-max_lookback):
    X_train.append(train[i:i+max_lookback-pred_interval])
    y_train.append(train[i+max_lookback,2])
X_train, y_train = np.array(X_train),np.array(y_train)
#X_train is now a 340 x 18 x 3 tensor which is what
inputs = df.iloc[len(df) - len(test) - 18:]
'''inputs = sc.transform(inputs)
X_test = []
for i in range(65):
    X_test.append(inputs[i:i+18,:])
X_test= np.array(X_test)
X_test.reshape(65,18,3)'''
