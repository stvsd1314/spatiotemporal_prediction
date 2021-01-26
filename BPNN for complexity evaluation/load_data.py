import csv
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn import preprocessing
import torch

def load_data():
    # dataset
    with open('D:/科研/碧月课题/代码/data/GY02.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    data=np.array(rows).astype(np.float64)#rows是数据类型是‘list',转化为数组类型好处理

    #读取label 转化成0 1 2
    with open('D:/科研/碧月课题/代码/data/GY02_status.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    data_label=np.array(rows).astype(np.float64)
    reconfactor=np.array([0,1,2],).astype(np.float64)
    data_label = np.dot(data_label,reconfactor)

    X_0 = data
    y_0 = data_label

    X_0 = preprocessing.StandardScaler().fit_transform(X_0)

    X = torch.Tensor(X_0).type(torch.FloatTensor)  # 格式很重要; tensor_size = [samples, features]
    y = torch.Tensor(y_0).type(torch.LongTensor)  # 格式很重要; tensor_size = [samples,]

    torch_dataset = Data.TensorDataset(X, y)
    train_size = int(0.75 * X.shape[0])
    test_size = X.shape[0] - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(torch_dataset, [train_size, test_size])

    return train_dataset, test_dataset

