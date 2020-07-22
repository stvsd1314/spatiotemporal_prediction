#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:31:43 2020

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
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras import optimizers
import numpy as np
import keras.backend as K
from Skip_LSTM import load_skip_data
from keras import Input
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import layers
from keras.models import Model

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

def load_spatio_data(): 
    # load dataset
    CD01_data = read_csv('/home/guotong/libiyue/additional/CD01.csv', header=None, index_col=None)
    CD01_label = read_csv('/home/guotong/libiyue/additional/CD01_status.csv', header=None, index_col=None)
    CD01_dataset = pd.concat([CD01_data, CD01_label], axis=1)
    
    CD02_data = read_csv('/home/guotong/libiyue/additional/CD02.csv', header=None, index_col=None)
    CD02_label = read_csv('/home/guotong/libiyue/additional/CD02_status.csv', header=None, index_col=None)
    CD02_dataset = pd.concat([CD02_data, CD02_label], axis=1)
    
    CD04_data = read_csv('/home/guotong/libiyue/additional/CD04.csv', header=None, index_col=None)
    CD04_label = read_csv('/home/guotong/libiyue/additional/CD04_status.csv', header=None, index_col=None)
    CD04_dataset = pd.concat([CD04_data, CD04_label], axis=1)
    
    GY01_data = read_csv('/home/guotong/libiyue/additional/GY01.csv', header=None, index_col=None)
    GY01_label = read_csv('/home/guotong/libiyue/additional/GY01_status.csv', header=None, index_col=None)
    GY01_dataset = pd.concat([GY01_data, GY01_label], axis=1)
    
    GY02_data = read_csv('/home/guotong/libiyue/additional/GY02.csv', header=None, index_col=None)
    GY02_label = read_csv('/home/guotong/libiyue/additional/GY02_status.csv', header=None, index_col=None)
    GY02_dataset = pd.concat([GY02_data, GY02_label], axis=1)
    
    KM03_data = read_csv('/home/guotong/libiyue/additional/KM03.csv', header=None, index_col=None)
    KM03_label = read_csv('/home/guotong/libiyue/additional/KM03_status.csv', header=None, index_col=None)
    KM03_dataset = pd.concat([KM03_data, KM03_label], axis=1)
    
    #CD01
    # dataset = pd.concat([ CD02_dataset, CD04_dataset, GY01_dataset, GY02_dataset, KM03_dataset, CD01_dataset], axis=1)
    
    # #CD02
    # dataset = pd.concat([CD01_dataset, CD04_dataset, GY01_dataset, GY02_dataset, KM03_dataset, CD02_dataset], axis=1)
    
    # #CD04
    dataset = pd.concat([CD01_dataset, GY01_dataset, GY02_dataset, KM03_dataset, CD02_dataset, CD04_dataset], axis=1)
    
    # #GY01
    # dataset = pd.concat([CD01_dataset, GY02_dataset, KM03_dataset, CD02_dataset,CD04_dataset, GY01_dataset], axis=1)
    
    # #GY02
    # dataset = pd.concat([KM03_dataset,CD01_dataset, CD02_dataset, CD04_dataset,  GY01_dataset, GY02_dataset], axis=1)
    
    # #KM03
    # dataset = pd.concat([CD01_dataset, CD04_dataset, GY01_dataset, GY02_dataset,CD02_dataset, KM03_dataset], axis=1)
    
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
    predict_length = 15
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_min, predict_length)
    # print(reframed.shape) # 448 = 16 * 28
     
    # split into train and test sets
    values = reframed.values
    n_train_hours = 650
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # split into input and outputs
    n_obs = n_min * n_features * 6
    train_X, train_y = train[:, :n_obs], train[:, -4:-1]
    test_X, test_y = test[:, :n_obs], test[:, -4:-1]
    # print(train_X.shape, len(train_X), train_y.shape)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_min, n_features*6))
    test_X = test_X.reshape((test_X.shape[0], n_min, n_features*6))
    
    return train_X, test_X


# load dataset
dataset = read_csv('/home/guotong/libiyue/additional/CD04.csv', header=None, index_col=None)
label = read_csv('/home/guotong/libiyue/additional/CD04_status.csv', header=None, index_col=None)
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
predict_length = 15

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

# load spatio data
spatio_train_x, spatio_test_x = load_spatio_data()


# the model
# input layer
input_tensor = Input(shape=(n_min, n_features))
input_tensor_skip = Input(shape=(14, n_features))
input_tensor_spatio = Input(shape=(n_min, n_features*6))


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

# spatio layer
s1 = Conv1D(filters=16, kernel_size=5, activation='relu')(input_tensor_spatio)
s2 = MaxPooling1D(pool_size=2)(s1)
# s1 = GRU(100, input_shape=(n_min, n_features*6))(input_tensor_spatio)
s2 = Dense(1, activation='tanh')(s2)
s2 = Flatten()(s2)

# concat
h1 = layers.concatenate([ls2, l2, c5, s2],axis=-1)
h2 = Dense(300, activation='tanh')(h1)
h3 = Dropout(0.3)(h2)
h4 = Dense(3, activation='softmax')(h3)

# define model 
model = Model([input_tensor, input_tensor_skip, input_tensor_spatio], h4)

# model compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# repeat 10 training
accuracy = []
for repeat in range(5):
    
    print('training : %d'%(repeat))
    # fit network
    history = model.fit([train_X, skip_train_x, spatio_train_x], train_y, epochs=20, batch_size=32, verbose=2, shuffle=False)

    acc = model.evaluate(x=[test_X[:94,:], skip_test_x[:94, :], spatio_test_x[:94, :]], y=test_y[:94,:], batch_size=5, verbose=1, sample_weight=None, steps=None)
    accuracy.append(acc[-1])
        
    print('accuracy is:{:.4f}, std is:{:.4f}'.format(np.mean(np.array(accuracy)), np.std(np.array(accuracy))))

print(np.max(accuracy))




















