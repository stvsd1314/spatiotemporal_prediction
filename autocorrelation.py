#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:40:46 2020

@author: guotong
"""
import pandas as pd
from pandas.plotting import autocorrelation_plot

data = pd.read_csv('CD01.csv', header=None, index_col=None)
data = pd.Series(data[25].values)
autocorrelation_plot(data)
# for i in range(28):
#     data = pd.Series(data[i].values)
#     autocorrelation_plot(data)
    


