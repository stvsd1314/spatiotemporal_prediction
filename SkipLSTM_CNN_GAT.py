#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:32:48 2020

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
from GAT_model import load_gnn_data
from keras_dgl.layers import GraphAttentionCNN

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
dataset = read_csv('/home/guotong/libiyue/additional/CD02.csv', header=None, index_col=None)
label = read_csv('/home/guotong/libiyue/additional/CD02_status.csv', header=None, index_col=None)
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

# load GAT data
X, Y_train, train_mask, A, graph_conv_filters, num_filters = load_gnn_data()

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

# GAT
input_tensor_gnn = Input(shape=(X.shape[1],))
x = Dropout(0.6, input_shape=(X.shape[1],))(input_tensor_gnn)
# g2 = GraphAttentionCNN(8, A, num_filters, graph_conv_filters, num_attention_heads=8, attention_combine='concat', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4))(x)
# g3 = Dropout(0.6)(g2)
# g4 = GraphAttentionCNN(Y.shape[1], A, num_filters, graph_conv_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4))(g3)
# g5 = Dense(50, activation='tanh')(g4)
# g6 = Dense(8, activation='tanh')(g5)
output_tensor_gnn = Dense(3, activation='softmax')(x)



# concat
h1 = layers.concatenate([ls2, l2, c5, x],axis=-1)
h2 = Dense(300, activation='tanh')(h1)
h3 = Dropout(0.3)(h2)
h4 = Dense(3, activation='softmax')(h3)

# define model 
model = Model([input_tensor, input_tensor_skip, input_tensor_gnn], [h4, output_tensor_gnn])

# model compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], loss_weights=[1.0, 0.1],)

# repeat 10 training
accuracy = []
for repeat in range(10):
    
    print('training : %d'%(repeat))
    # fit network
    history = model.fit([train_X, skip_train_x, X], [train_y, Y_train] , epochs=20, batch_size=A.shape[0], verbose=2, shuffle=False)

    acc = model.evaluate(x=[test_X[:94,:], skip_test_x[:94, :], X[:94, :]], y=[test_y[:94,:], test_y[:94,:]], batch_size=5, verbose=1, sample_weight=None, steps=None)
    accuracy.append(acc[-1])
        
    print('accuracy is:{:.4f}, std is:{:.4f}'.format(np.mean(np.array(accuracy)), np.std(np.array(accuracy))))

print(np.max(accuracy))










