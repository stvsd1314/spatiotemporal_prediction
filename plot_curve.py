#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 22:04:13 2020

@author: guotong
"""


import numpy as np
import matplotlib.pyplot as plt

# CD01
GAT_SkipLSTM_CNN = np.array([0.9574, 0.8510, 0.7446, 0.7276])
GAT_SkipLSTM_CNN_std = np.array([0.016,0.0741,0.1383,0.0532])
GAT_SkipLSTM_CNN_upper = GAT_SkipLSTM_CNN + GAT_SkipLSTM_CNN_std
GAT_SkipLSTM_CNN_lower = GAT_SkipLSTM_CNN - GAT_SkipLSTM_CNN_std

 
without_GAT = np.array([0.9255,0.7766,0.6915,0.6702])
without_GAT_std = np.array([0.0431,0.0357,0.0517,0.0698])
without_GAT_upper = without_GAT + without_GAT_std
without_GAT_lower = without_GAT - without_GAT_std

without_SkipLSTM = np.array([0.8327,0.7071,0.6659,0.6018])
without_SkipLSTM_std = np.array([0.1414,0.0988,0.0361,0.0687])
without_SkipLSTM_upper = without_SkipLSTM + without_SkipLSTM_std 
without_SkipLSTM_lower= without_SkipLSTM - without_SkipLSTM_std 

without_CNN = np.array([0.9212, 0.8, 0.6489, 0.6297])
without_CNN_std = np.array([0.032,0.0352,0.0671,0.0473])
without_CNN_upper = without_CNN + without_CNN_std
without_CNN_lower = without_CNN - without_CNN_std

# 1_std = 
# 5_std = 
# 10_std = 
# 15_std = 

fig, ax1 = plt.subplots()

x = np.array([1,5,10,15])


ax1.plot(x,GAT_SkipLSTM_CNN,marker='s', color='r',markersize=4,linewidth=1,label='GAT SkipLSTM CNN')
ax1.plot(x,without_GAT,marker='s',color='b',markersize=4,linewidth=1,label='w/o GAT')
ax1.plot(x,without_SkipLSTM ,marker='s',color='g',markersize=4,linewidth=1,label='w/o Skip')
ax1.plot(x,without_CNN ,marker='s',color='purple',markersize=4,linewidth=1,label='w/o CNN')
ax1.set_ylim(0,1)

# ax.fill_between(x,GAT_SkipLSTM_CNN_upper,GAT_SkipLSTM_CNN_lower,color='r',alpha=0.1)
# ax.fill_between(x,without_GAT_upper,without_GAT_lower,color='b',alpha=0.1)
# ax.fill_between(x,without_SkipLSTM_upper,without_SkipLSTM_lower,color='g',alpha=0.1)
# ax.fill_between(x,without_CNN_upper,without_CNN_lower,color='purple',alpha=0.1)

ax2 = ax1.twinx()
ax2.bar(x, GAT_SkipLSTM_CNN_std, color='r')
ax2.bar(x, without_GAT_std, color='b')
ax2.bar(x, without_SkipLSTM_std, color='g')
ax2.bar(x, without_CNN_std, color='purple')


ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Vriance')
ax2.set_ylim(0, 0.5)



 
plt.xlabel('Time for predicted')
# plt.ylabel()
plt.legend(fontsize=6)
plt.title(' Accuracy of CD01')

plt.grid(axis='y',linestyle='-.')
plt.show()


