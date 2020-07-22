#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:31:22 2020

@author: guotong
"""

# import matplotlib.pyplot as plt
# import numpy as np

# index = np.arange(5)
# values = [5, 6, 3, 4, 6]
# SD = [0.8, 2, 0.4, 0.9, 1.3]
# plt.title('A Bar Chart')
# plt.bar(index, values, yerr = SD, error_kw = {'ecolor' : '0.2', 'capsize' :6}, alpha=0.7, label = 'First')
# plt.xticks(index+0.2,['a', 'b', 'c', 'd', 'e'])
# plt.legend(loc=2)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

#%%
# n_groups = 4 # 4个时间长度
# gat_skiplstm_cnn = (0.9574, 0.8510, 0.7446, 0.7276)
# gat_skiplstm_cnn_std = (0.016,0.0741,0.1383,0.0532)

# without_gat = (0.9255,0.7766,0.6915,0.6702)
# without_gat_std = (0.0431,0.0357,0.0517,0.0698)

# without_skiplstm = (0.8327,0.7071,0.6659,0.6018)
# without_skiplstm_std = (0.1414,0.0988,0.0361,0.0687)

# without_cnn = (0.9212,0.8,0.6489,0.6297)
# without_cnn_std = (0.032,0.0352,0.0671,0.0473)


# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.15

# opacity = 0.4
# error_config = {'ecolor': '0.2', 'capsize' :2}

# rects1 = ax.bar(index, gat_skiplstm_cnn, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=gat_skiplstm_cnn_std, error_kw=error_config,
#                 label='GSC')

# rects2 = ax.bar(index + bar_width, without_gat , bar_width,
#                 alpha=opacity, color='r',
#                 yerr=without_gat_std, error_kw=error_config,
#                 label='GSC w/o GAT')

# rects3 = ax.bar(index + bar_width*2, without_skiplstm , bar_width,
#                 alpha=opacity, color='y',
#                 yerr=without_skiplstm_std, error_kw=error_config,
#                 label='GSC w/o CNN')

# rects4 = ax.bar(index + bar_width*3, without_cnn , bar_width,
#                 alpha=opacity, color='g',
#                 yerr=without_cnn_std, error_kw=error_config,
#                 label='GSC w/o SkipLSTM')
# plt.style.use('seaborn-darkgrid')
# ax.set_ylabel('Accuracy', fontsize=15)
# ax.set_xlabel('Prediction length', fontsize=15)
# ax.set_title('Ablation experiments on CD01', fontsize=15)
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('1', '5', '10', '15'))
# ax.legend()

# fig.tight_layout()
# plt.show()

#%%
# n_groups = 4 # 4个时间长度
# gat_skiplstm_cnn = (0.9212,0.7978,0.7111,0.6013 )
# gat_skiplstm_cnn_std = (0.0173,0.0527,0.1044,0.0781) 

# without_gat = (0.8111,0.7128,0.6383,0.6002)
# without_gat_std = (0.0244,0.0415,0.0664,0.0455 )

# without_skiplstm = (0.7892,0.6159,0.5830,0.4915 )
# without_skiplstm_std = (0.0462,0.0218,0.0544,0.0594 )

# without_cnn = ( 0.8851,0.6765,0.5957,0.5638)
# without_cnn_std = ( 0.04,0.0868,0.1249,0.0494)

 
# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.15

# opacity = 0.4
# error_config = {'ecolor': '0.2', 'capsize' :2}

# rects1 = ax.bar(index, gat_skiplstm_cnn, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=gat_skiplstm_cnn_std, error_kw=error_config,
#                 label='GSC')

# rects2 = ax.bar(index + bar_width, without_gat , bar_width,
#                 alpha=opacity, color='r',
#                 yerr=without_gat_std, error_kw=error_config,
#                 label='GSC w/o GAT')

# rects3 = ax.bar(index + bar_width*2, without_skiplstm , bar_width,
#                 alpha=opacity, color='y',
#                 yerr=without_skiplstm_std, error_kw=error_config,
#                 label='GSC w/o CNN')

# rects4 = ax.bar(index + bar_width*3, without_cnn , bar_width,
#                 alpha=opacity, color='g',
#                 yerr=without_cnn_std, error_kw=error_config,
#                 label='GSC w/o SkipLSTM')
# plt.style.use('seaborn-darkgrid')
# ax.set_ylabel('Accuracy', fontsize=15)
# ax.set_xlabel('Prediction length', fontsize=15)
# ax.set_title('Ablation experiments on CD04', fontsize=15)
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('1', '5', '10', '15'))
# ax.legend()

# fig.tight_layout()
# plt.show()

#%%
n_groups = 4 # 4个时间长度
gat_skiplstm_cnn = (0.955,0.8844,0.6777,0.6555 )
gat_skiplstm_cnn_std = (0.0191,0.0311,0.0761,0.0163 ) 

without_gat = (0.9255,0.8085,0.616,0.5809 )
without_gat_std = (0.0178,0.0353,0.0675,0.0519 )

without_skiplstm = ( 0.8428,0.7246,0.5621,0.5276)
without_skiplstm_std = (0.0405,0.0803,0.0709,0.0587  )

without_cnn = ( 0.9191,0.617,0.634,0.6382 )
without_cnn_std = (0.029,0.0311,0.0347,0.0301 )
 
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.4
error_config = {'ecolor': '0.2', 'capsize' :2}

rects1 = ax.bar(index, gat_skiplstm_cnn, bar_width,
                alpha=opacity, color='b',
                yerr=gat_skiplstm_cnn_std, error_kw=error_config,
                label='GSC')

rects2 = ax.bar(index + bar_width, without_gat , bar_width,
                alpha=opacity, color='r',
                yerr=without_gat_std, error_kw=error_config,
                label='GSC w/o GAT')

rects3 = ax.bar(index + bar_width*2, without_skiplstm , bar_width,
                alpha=opacity, color='y',
                yerr=without_skiplstm_std, error_kw=error_config,
                label='GSC w/o CNN')

rects4 = ax.bar(index + bar_width*3, without_cnn , bar_width,
                alpha=opacity, color='g',
                yerr=without_cnn_std, error_kw=error_config,
                label='GSC w/o SkipLSTM')
plt.style.use('seaborn-darkgrid')
ax.set_ylabel('Accuracy', fontsize=15)
ax.set_xlabel('Prediction length', fontsize=15)
ax.set_title('Ablation experiments on GY01', fontsize=15)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '5', '10', '15'))
ax.legend()

fig.tight_layout()
plt.show()
#%%
# n_groups = 4 # 4个时间长度
# gat_skiplstm_cnn = (0.8,0.6444,0.6177,0.5888 )
# gat_skiplstm_cnn_std = (0.0581,0.0229,0.0249,0.0253  ) 

# without_gat = ( 0.7617,0.6382,0.6276,0.5851 )
# without_gat_std = (0.0415,0.0353,0.0335,0.0274  )

# without_skiplstm = ( 0.5321,0.4818,0.4686,0.4699 )
# without_skiplstm_std = ( 0.0488,0.0334,0.0386,0.0295 )

# without_cnn = ( 0.6595,0.634,0.617,0.5425  )
# without_cnn_std = (0.0359,0.0401,0.017,0.0213 )


 

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.15

# opacity = 0.4
# error_config = {'ecolor': '0.2', 'capsize' :2}

# rects1 = ax.bar(index, gat_skiplstm_cnn, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=gat_skiplstm_cnn_std, error_kw=error_config,
#                 label='GSC')

# rects2 = ax.bar(index + bar_width, without_gat , bar_width,
#                 alpha=opacity, color='r',
#                 yerr=without_gat_std, error_kw=error_config,
#                 label='GSC w/o GAT')

# rects3 = ax.bar(index + bar_width*2, without_skiplstm , bar_width,
#                 alpha=opacity, color='y',
#                 yerr=without_skiplstm_std, error_kw=error_config,
#                 label='GSC w/o CNN')

# rects4 = ax.bar(index + bar_width*3, without_cnn , bar_width,
#                 alpha=opacity, color='g',
#                 yerr=without_cnn_std, error_kw=error_config,
#                 label='GSC w/o SkipLSTM')
# plt.style.use('seaborn-darkgrid')
# ax.set_ylabel('Accuracy', fontsize=15)
# ax.set_xlabel('Prediction length', fontsize=15)
# ax.set_title('Ablation experiments on GY02', fontsize=15)
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('1', '5', '10', '15'))
# ax.legend()

# fig.tight_layout()
# plt.show()

#%%
# n_groups = 4 # 4个时间长度
# gat_skiplstm_cnn = (0.7553,0.7234,0.6702,0.6489 )
# gat_skiplstm_cnn_std = (0.0163,0.1497,0.1254,0.0919 ) 

# without_gat = ( 0.7127,0.7064,0.6457,0.6064)
# without_gat_std = (0.0701,0.0313,0.029,0.0904  )

# without_skiplstm = ( 0.6321,0.576,0.5166,0.515 )
# without_skiplstm_std = ( 0.0685,0.0611,0.0711,0.0599 )

# without_cnn = ( 0.7425,0.6702,0.6276,0.5808 )
# without_cnn_std = ( 0.0289,0.033,0.049,0.0801 )


 

# fig, ax = plt.subplots()

# index = np.arange(n_groups)
# bar_width = 0.15

# opacity = 0.4
# error_config = {'ecolor': '0.2', 'capsize' :2}

# rects1 = ax.bar(index, gat_skiplstm_cnn, bar_width,
#                 alpha=opacity, color='b',
#                 yerr=gat_skiplstm_cnn_std, error_kw=error_config,
#                 label='GSC')

# rects2 = ax.bar(index + bar_width, without_gat , bar_width,
#                 alpha=opacity, color='r',
#                 yerr=without_gat_std, error_kw=error_config,
#                 label='GSC w/o GAT')

# rects3 = ax.bar(index + bar_width*2, without_skiplstm , bar_width,
#                 alpha=opacity, color='y',
#                 yerr=without_skiplstm_std, error_kw=error_config,
#                 label='GSC w/o CNN')

# rects4 = ax.bar(index + bar_width*3, without_cnn , bar_width,
#                 alpha=opacity, color='g',
#                 yerr=without_cnn_std, error_kw=error_config,
#                 label='GSC w/o SkipLSTM')
# plt.style.use('seaborn-darkgrid')
# ax.set_ylabel('Accuracy', fontsize=15)
# ax.set_xlabel('Prediction length', fontsize=15)
# ax.set_title('Ablation experiments on KM03', fontsize=15)
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(('1', '5', '10', '15'))
# ax.legend()

# fig.tight_layout()
# plt.show()









