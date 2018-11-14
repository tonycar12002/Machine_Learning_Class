# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_train = pd.read_csv('train.csv', header=None)
dataset_test = pd.read_csv('test.csv', header=None)


# Take all columns except last one
X_train = dataset_train.iloc[:, :-1]
X_test = dataset_test.iloc[:, :]
y_train = dataset_train.iloc[:, dataset_train.shape[1]-1]


# Encoding categorial data
# Replace string by categories number
drop = []
train_objs_num = len(X_train)

dataset = pd.concat(objs=[X_train, X_test], axis=0)

for row in range(0, X_train.shape[1]):
    if (isinstance(X_train.values[1][row], str)):        
        print (row), 
        one_hot = pd.get_dummies(dataset[row])
        drop.append(row)
        dataset = pd.concat([dataset, one_hot], axis=1)
        dataset = dataset.iloc[:, :-1]
        

# Remove original attributes
drop.sort(reverse=True)
for row in drop:
    dataset = dataset.drop(row, axis = 1)


X_train = dataset[:train_objs_num]
X_test = dataset[train_objs_num:]
