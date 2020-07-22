#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:38:40 2020

@author: guotong
"""

from keras.layers import Conv1D, Dense, MaxPool1D, concatenate, Flatten
from keras import Input, Model
from keras.utils import plot_model
import numpy as np
 
 
def multi_input_model():
    """构建多输入模型"""
    input1_= Input(shape=(100, 1), name='input1')
    input2_ = Input(shape=(50, 1), name='input2')
 
    x1 = Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input1_)
    x1 = MaxPool1D(pool_size=10, strides=10)(x1)
 
    x2 = Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input2_)
    x2 = MaxPool1D(pool_size=5, strides=5)(x2)
    
    print(x1.shape, x2.shape)
    
    x = concatenate([x1, x2])
    x = Flatten()(x)
 
    x = Dense(10, activation='relu')(x)
    output_ = Dense(1, activation='sigmoid', name='output')(x)
 
    model = Model(inputs=[input1_, input2_], outputs=[output_]) 
    return model
 
if __name__ == '__main__':
    # 产生训练数据
    x1 = np.random.rand(100, 100, 1)
    x2 = np.random.rand(100, 50, 1)
    # 产生标签
    y = np.random.randint(0, 2, (100,))
 
    model = multi_input_model()
 
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit([x1, x2], y, epochs=10, batch_size=10)
 
    
 
    
 
    
 
    