#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:13:46 2022

@author: ragharao
"""
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

directory = '/Users/ragharao/Desktop/CNN/AAPL.csv'

data = pd.read_csv(directory)

training = StandardScaler().fit_transform(data.iloc[:, 2:3].values)



n = len(training)


training_x = np.array([training[i-50:i] for i in range(50,n)])
training_y = np.array([training[i] for i in range(50,n)])



testing_x = training_x[870:]
testing_y = training_y[870:]

training_x = training_x[:870]
training_y = training_y[:870]



model = Sequential()
model.add(LSTM(units = 30, return_sequences = True, input_shape = (training_x.shape[1], 1)))
model.add(LSTM(units = 30, return_sequences = True))
model.add(LSTM(units = 30))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(training_x, training_y, epochs = 50, batch_size = 32)


y_pred = model.predict(testing_x)
plt.plot(testing_y, color = 'green', label = 'Real Apple Stock Price',ls='--')
plt.plot(y_pred, color = 'red', label = 'Predicted Apple Stock Price',ls='-')
plt.title('Predicted Stock Price')
plt.xlabel('Time in days')
plt.ylabel('Real Stock Price')
plt.legend()
plt.show()




