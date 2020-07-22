#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:21:36 2020

@author: guotong
"""

import pandas as pd
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import numpy as np

# convert series to supervised learning
def series_to_supervised_with_skip(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -15):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def load_skip_data():
    # load dataset
    dataset = read_csv('/home/guotong/libiyue/additional/CD01.csv', header=None, index_col=None)
    label = read_csv('/home/guotong/libiyue/additional/CD01_status.csv', header=None, index_col=None)
    dataset = pd.concat([dataset, label], axis=1)
    
    values = dataset.values
    
    # ensure all data is float
    values = values.astype('float32')
    
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # specify the number of lag hours
    n_min = 200
    n_features = 31
    
    time_step = n_min // 15 + 1 # 14
    
    # the number of predicted samples
    predict_length = 15
    
    # frame as supervised learning
    reframed = series_to_supervised_with_skip(scaled, n_min, predict_length)
    # print(reframed.shape) # 448 = 16 * 28
     
    # split into train and test sets
    values = reframed.values
    n_train_hours = 650
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # split into input and outputs
    n_obs = time_step * n_features
    
    train_X, train_y = train[:, :n_obs], train[:, -4:-1]
    test_X, test_y = test[:, :n_obs], test[:, -4:-1]
    # print(train_X.shape, len(train_X), train_y.shape)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], time_step, n_features))
    test_X = test_X.reshape((test_X.shape[0], time_step, n_features))
    
    return train_X, train_y, test_X, test_y

# # design network
# model = Sequential()
# model.add(LSTM(200, input_shape=(time_step, n_features)))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='tanh'))
# model.add(Dropout(0.3))
# model.add(Dense(3, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=128, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# print('\n')    
# print('accuracy is:{:.4f}, std is:{:.4f}'.format(np.mean(np.array(history.history['val_acc'])), np.std(np.array(history.history['val_acc']))))
# print('\n')
 


# #plot history
# # pyplot.plot(history.history['loss'], label='train')
# # pyplot.plot(history.history['val_loss'], label='test')
# pyplot.plot(history.history['acc'], label='train')
# pyplot.plot(history.history['val_acc'], label='test')
# # pyplot.legend()