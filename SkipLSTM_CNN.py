#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:53:04 2020

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
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import numpy as np
from keras import layers
from keras.models import Model
from keras import Input
from Skip_LSTM import load_skip_data

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
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
 
# load dataset
dataset = read_csv('/home/guotong/libiyue/additional/KM03.csv', header=None, index_col=None)
label = read_csv('/home/guotong/libiyue/additional/KM03_status.csv', header=None, index_col=None)
dataset = pd.concat([dataset, label], axis=1)

values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# specify the number of lag hours
n_min = 30
n_features = 31

# the number of predicted samples
predict_length = 1

# frame as supervised learning
reframed = series_to_supervised(scaled, n_min, predict_length)
# print(reframed.shape) # 448 = 16 * 28
 
# split into train and test sets
values = reframed.values
n_train_hours = 650
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_min * n_features
train_X, train_y = train[:, :n_obs], train[:, -4:-1]
test_X, test_y = test[:, :n_obs], test[:, -4:-1]
# print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_min, n_features))
test_X = test_X.reshape((test_X.shape[0], n_min, n_features))

# load skip data
skip_train_x, skip_train_y, skip_test_x, skip_test_y = load_skip_data()

# the model
# input layer
input_tensor = Input(shape=(n_min, n_features))
input_tensor_skip = Input(shape=(14, n_features))

# Skip LSTM
ls1 = LSTM(200)(input_tensor_skip)
ls2 = Dropout(0.3)(ls1)

# LSTM
l1 = LSTM(200)(input_tensor) # (batch, n_min*n_features)
l2 = Dropout(0.3)(l1)

# CNN
c1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_tensor)
c2 = Conv1D(filters=64, kernel_size=5, activation='relu')(c1)
c3 = Dropout(0.5)(c2)
c4 = MaxPooling1D(pool_size=2)(c3)
c5 = Flatten()(c4)
 
# concat
h1 = layers.concatenate([ls2, l2, c5],axis=-1)
h2 = Dense(300, activation='tanh')(h1)
h3 = Dropout(0.3)(h2)
h4 = Dense(3, activation='softmax')(h3)

# define model 
model = Model([input_tensor, input_tensor_skip], h4)

# model compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# repeat 10 training
accuracy = []
for repeat in range(4):
    
    print('training : %d'%(repeat))
    # fit network
    history = model.fit([train_X, skip_train_x], train_y, epochs=20, batch_size=32, verbose=2, shuffle=False)

    acc = model.evaluate(x=[test_X[:94,:], skip_test_x[:94, :]], y=test_y[:94,:], batch_size=5, verbose=1, sample_weight=None, steps=None)
    accuracy.append(acc[-1])
        
    print('accuracy is:{:.4f}, std is:{:.4f}'.format(np.mean(np.array(accuracy)), np.std(np.array(accuracy))))

print(np.max(accuracy))



