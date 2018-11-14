# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Importing the dataset
dataset_train = pd.read_csv('train.csv', header=None)
dataset_test = pd.read_csv('test.csv', header=None)


# Take all columns except last one
X_train = dataset_train.iloc[:, :-1]
X_test = dataset_test.iloc[:, :]
y = dataset_train.iloc[:, dataset_train.shape[1]-1]


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

X = dataset[:train_objs_num]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature normalize
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='rbf')
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)


print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)



test_data = dataset[train_objs_num:]
test_std = sc.transform(test_data)
test_std_pred = svm.predict(test_std)

import csv
with open('/output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'and'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])


