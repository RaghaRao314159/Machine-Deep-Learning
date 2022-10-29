#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:41:37 2022

@author: ragharao
"""
#many libraries

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

#===============================================================
#Prepare data for processing

directory = '/Users/ragharao/Desktop/diabetes.csv'

diabetes = pd.read_csv(directory)


diabetes_x = diabetes.iloc[:,:-1]
diabetes_y = diabetes.iloc[:,-1]

columns = diabetes_x.columns.values

for i in columns:
    maxi = max(diabetes_x[i])
    mini = min(diabetes_x[i])
    
    diabetes_x[i] = [(j - mini)/(maxi - mini) for j in diabetes_x[i]]

print(diabetes_x.shape)
#(768, 8)


#===============================================================
#Determine ideal number of epochs to avoid under/overfitting



model_test = Sequential()
model_test.add(Dense(64,activation='tanh',input_dim=8))
model_test.add(Dense(32,activation='tanh'))
model_test.add(Dense(16,activation='tanh'))
model_test.add(Dense(1,activation='sigmoid'))
model_test.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
graph = model_test.fit(x=diabetes_x,y=diabetes_y,batch_size=4,validation_split=0.2,epochs=70,verbose=1)


k = graph.history['val_loss']
print(np.argmin(k))
#41

plt.plot(graph.history['loss'])
plt.plot(k)


# The loss/accuracy is a minimum/maximum at 40 epochs so we shall use 50 epochs to test our models

#===============================================================
#Now let's test difefrent model architecture

def model_1():
  model = Sequential()
  model.add(Dense(64,activation='tanh',input_dim=8))
  model.add(Dense(32,activation='tanh'))
  model.add(Dense(16,activation='tanh'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
  return model

model_64= KerasRegressor(build_fn=model_1, epochs = 50, batch_size=4, verbose=1)

score = cross_val_score(estimator=model_64,X=diabetes_x, y=diabetes_y, cv=5)

print(abs(score.mean()))

#average loss = 0.16516617486883944


#---------------------------------------------------------------

def model_2():
  model = Sequential()
  model.add(Dense(32,activation='tanh',input_dim=8))
  model.add(Dense(16,activation='tanh'))
  model.add(Dense(8,activation='tanh'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
  return model

model_32= KerasRegressor(build_fn=model_2, epochs = 50, batch_size=4, verbose=1)

score = cross_val_score(estimator=model_32,X=diabetes_x, y=diabetes_y, cv=5)

print(abs(score.mean()))

#average loss = 0.1567052008315994

#---------------------------------------------------------------

def model_3():
  model = Sequential()
  model.add(Dense(16,activation='tanh',input_dim=8))
  model.add(Dense(8,activation='tanh'))
  model.add(Dense(4,activation='tanh'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
  return model

model_16= KerasRegressor(build_fn=model_3, epochs = 50, batch_size=4, verbose=1)

score = cross_val_score(estimator=model_16,X=diabetes_x, y=diabetes_y, cv=5)

print(abs(score.mean()))

#average loss = 0.15630184466088207

#---------------------------------------------------------------

def model_4():
  model = Sequential()
  model.add(Dense(8,activation='tanh',input_dim=8))
  model.add(Dense(4,activation='tanh'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
  return model

model_8= KerasRegressor(build_fn=model_4, epochs = 50, batch_size=4, verbose=1)

score = cross_val_score(estimator=model_8,X=diabetes_x, y=diabetes_y, cv=5)

print(abs(score.mean()))

#average loss = 0.1575376131956501

#===============================================================

'''
best model is model_3
'''
