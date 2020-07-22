#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:08:56 2020

@author: guotong
"""
from keras.models import load_model
import pandas as pd
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.layers import LSTM, Dense, Activation, Dropout
from keras import Input
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from utils import *
from keras_dgl.layers import GraphCNN

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

def load_data(data_filepath=None, label_filepath=None):
    # load dataset
    dataset = read_csv(data_filepath, header=None, index_col=None)
    label = read_csv(label_filepath, header=None, index_col=None)

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
    
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_min, 1)
    print(reframed.shape) # 448 = 16 * 28
     
    # split into train and test sets
    values = reframed.values
    # n_train_hours = 650
    # train = values[:n_train_hours, :]
    # test = values[n_train_hours:, :]
    
    # split into input and outputs
    n_obs = n_min * n_features
    # train_X, train_y = train[:, :n_obs], train[:, -4:-1]
    # test_X, test_y = test[:, :n_obs], test[:, -4:-1]
    # print(train_X.shape, len(train_X), train_y.shape)
    
    X, y = values[:, :n_obs], values[:, -4:-1]
    
    # # reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((train_X.shape[0], n_min, n_features))
    # test_X = test_X.reshape((test_X.shape[0], n_min, n_features))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    # return train_X, train_y, test_X, test_y
    return X, y

def get_data():
    # CD01_train_x, CD01_train_y, CD01_test_x, CD01_test_y = load_data('/home/guotong/libiyue/additional/CD01.csv', '/home/guotong/libiyue/additional/CD01_status.csv')
    # CD02_train_x, CD02_train_y, CD02_test_x, CD02_test_y = load_data('/home/guotong/libiyue/additional/CD02.csv', '/home/guotong/libiyue/additional/CD02_status.csv')
    # CD04_train_x, CD04_train_y, CD04_test_x, CD04_test_y = load_data('/home/guotong/libiyue/additional/CD04.csv', '/home/guotong/libiyue/additional/CD04_status.csv')
    # GY01_train_x, GY01_train_y, GY01_test_x, GY01_test_y = load_data('/home/guotong/libiyue/additional/GY01.csv', '/home/guotong/libiyue/additional/GY01_status.csv')
    # GY02_train_x, GY02_train_y, GY02_test_x, GY02_test_y = load_data('/home/guotong/libiyue/additional/GY02.csv', '/home/guotong/libiyue/additional/GY02_status.csv')
    # KM03_train_x, KM03_train_y, KM03_test_x, KM03_test_y = load_data('/home/guotong/libiyue/additional/KM03.csv', '/home/guotong/libiyue/additional/KM03_status.csv')
    
    # train_x = np.concatenate((CD01_train_x, CD02_train_x, CD04_train_x, GY01_train_x, GY02_train_x, KM03_train_x), axis=0)
    # train_y = np.concatenate((CD01_train_y, CD02_train_y, CD04_train_y, GY01_train_y, GY02_train_y, KM03_train_y), axis=0)
    
    # test_x = np.concatenate((CD01_test_x, CD02_test_x, CD04_test_x, GY01_test_x, GY02_test_x, KM03_test_x), axis=0)
    # test_y = np.concatenate((CD01_test_y, CD02_test_y, CD04_test_y, GY01_test_y, GY02_test_y, KM03_test_y), axis=0)
    
    X_CD01, y_CD01 = load_data('/home/guotong/libiyue/additional/CD01.csv', '/home/guotong/libiyue/additional/CD01_status.csv')
    X_CD02, y_CD02 = load_data('/home/guotong/libiyue/additional/CD02.csv', '/home/guotong/libiyue/additional/CD02_status.csv')
    X_CD04, y_CD04 = load_data('/home/guotong/libiyue/additional/CD04.csv', '/home/guotong/libiyue/additional/CD04_status.csv')
    X_GY01, y_GY01 = load_data('/home/guotong/libiyue/additional/GY01.csv', '/home/guotong/libiyue/additional/GY01_status.csv')
    X_GY02, y_GY02 = load_data('/home/guotong/libiyue/additional/GY02.csv', '/home/guotong/libiyue/additional/GY02_status.csv')
    X_KM03, y_KM03 = load_data('/home/guotong/libiyue/additional/KM03.csv', '/home/guotong/libiyue/additional/KM03_status.csv')
    
    X = np.concatenate((X_CD01, X_CD02, X_CD04, X_GY01, X_GY02, X_KM03), axis=0)
    y = np.concatenate((y_CD01, y_CD02, y_CD04, y_GY01, y_GY02, y_KM03), axis=0)
    
    # return train_x, train_y, test_x, test_y
    return X, y

def get_adj_matrix(x):
    adj_matrix = np.zeros((x.shape[0], x.shape[0]))
    adj_matrix[0:650, 0:2600] = 1     # CD01
    adj_matrix[650:1300, :] = 1       # CD02
    adj_matrix[1300:1950, 0:1950] = 1 # CD04
    adj_matrix[1950:2600, 0:1300] = 1 # GY01
    adj_matrix[1950:2600, 1950:3250] = 1
    adj_matrix[2600:3250, 650:1300] = 1 # GY02
    adj_matrix[2600:3250, 1950:3900] = 1
    adj_matrix[3250:3900, 650:1300] = 1 # KM03
    adj_matrix[3250:3900, 2600:3900]
    
    return adj_matrix



def load_gnn_model():
    
    X, Y = get_data()
    A = get_adj_matrix(X)
    
    _, Y_val, _, train_idx, val_idx, test_idx, train_mask = get_splits(Y)
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    labels = np.argmax(Y, axis=1) + 1
    
    Y_train = np.zeros(Y.shape)
    labels_train = np.zeros(labels.shape)
    Y_train[train_idx] = Y[train_idx]
    labels_train[train_idx] = labels[train_idx]
    
    Y_test = np.zeros(Y.shape)
    labels_test = np.zeros(labels.shape)
    Y_test[test_idx] = Y[test_idx]
    labels_test[test_idx] = labels[test_idx]
    
    # Build Graph Convolution filters
    SYM_NORM = True
    # A_norm = preprocess_adj_numpy(A, SYM_NORM)
    A_norm = A
    num_filters = 2
    graph_conv_filters = np.concatenate([A_norm, np.matmul(A_norm, A_norm)], axis=0)
    graph_conv_filters = K.constant(graph_conv_filters)
    
    # build GAT model
    input_tensor = Input(shape=(X.shape[1],))
    x = Dropout(0.6, input_shape=(X.shape[1],))(x)
    model.add(GraphAttentionCNN(8, A, num_filters, graph_conv_filters, num_attention_heads=8, attention_combine='concat', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
    model.add(Dropout(0.6))
    model.add(GraphAttentionCNN(Y.shape[1], A, num_filters, graph_conv_filters, num_attention_heads=1, attention_combine='average', attention_dropout=0.6, activation='elu', kernel_regularizer=l2(5e-4)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])
    
    # Build Model
    input_tensor = Input(shape=(X.shape[1],))
    x = GraphCNN(16, num_filters, graph_conv_filters, input_shape=(X.shape[1],), activation='elu', kernel_regularizer=l2(5e-4))(input_tensor)
    x = Dropout(0.2)(x)
    x = GraphCNN(16, num_filters, graph_conv_filters, activation='elu', kernel_regularizer=l2(5e-4), name='g_out')(x)
    
    output_tensor = Dense(Y.shape[1], activation='softmax', name='sub_out')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])
    model.summary()
    
    # load
    model.load_weights('gnn.h5')
    
    # # Y_pred = model.predict(X, batch_size=A.shape[0])
    
    sub_model = Model(inputs = model.input, outputs = model.get_layer('g_out').output )
    
    # result = sub_model.predict(X, batch_size=A.shape[0])
     
    # print(result)
    
    return sub_model



m = load_gnn_model()
m.predict(X, batch_size=A.shape[0])











