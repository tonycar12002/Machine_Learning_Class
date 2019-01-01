#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import json
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 


# In[2]:


train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

X = train_dataset.iloc[:, 5]


X_ = test_dataset.iloc[:, 4]

dataset_text = X.append(X_)

train_objs_num = len(X)
print(train_objs_num)
print(dataset_text.shape)


# In[3]:


# Data preprocessing
data_pre = dataset_text
data_pre = data_pre.apply(lambda x:x.lower())
data_pre = data_pre.apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x)) #Remove all characters not a~z and 0~9

tokenizer = Tokenizer(num_words = 2500, split=' ')
tokenizer.fit_on_texts(data_pre.values)
data_pre = tokenizer.texts_to_sequences(data_pre.values)
data_pre = pad_sequences(data_pre)

print("Done")


# In[4]:


print(data_pre.shape)
print(train_objs_num)

X = data_pre[:train_objs_num]
test = data_pre[train_objs_num:]

y = train_dataset.iloc[:, 0]

print(X.shape)
print(y.shape)
print(test.shape)
X


# In[10]:


embed_dim = 128
lstm_out = 350
batch_size= 1000

model = Sequential()
model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout=0.1))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

Y = pd.get_dummies(y).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.20, random_state = 36)

#Here we train the Network.

model.fit(X_train, Y_train, batch_size = batch_size, epochs = 10,  verbose = 1)

# Measuring score and accuracy on validation set

score,acc = model.evaluate(X_valid, Y_valid, verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))


# In[11]:


pred = model.predict(test, verbose=1)


# In[7]:


pred
Y_valid


# In[12]:


with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Sentiment'])
    for i in range(len(pred)):
        writer.writerow([i, pred[i][1]*4])

print ("Done")

