
# coding: utf-8

# ## Import Libraries 

# In[1]:


# Import Lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from xgboost import XGBClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
import math
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

# Importing the dataset
dataset_train = pd.read_csv('train.csv', header=None)
dataset_test = pd.read_csv('test.csv', header=None)
header = ['Age','Workclass','fnlwgt','Education','Education-num','Marital-status','Occupation','Relationship','Race','Sex','Capital-gain','Capital-loss','Hours-per-week','Native-country','Income']
dataset_train = pd.DataFrame(dataset_train.values, columns = header) 
dataset_test = pd.DataFrame(dataset_test.values, columns = header[:-1]) 

# Take all columns except last one
train = dataset_train.iloc[:, :-1]
test = dataset_test.iloc[:, :]
y = dataset_train.iloc[:, dataset_train.shape[1]-1]
y = y.astype(int)
print(dataset_train.shape)


# ## Data Analysis

# In[42]:


dataset_train


# In[43]:


matplotlib.rcParams.update({'xtick.labelsize': 18})
#print(list(dataset_train)[0])
fig = plt.figure(figsize=(25,100))
fig.subplots_adjust(hspace = 1.5)
now = 1
for i in range(14):
    if i!=2 and i!=10 and i!=11:
        plt.subplot(14, 1, now)
        now += 1
        key = list(dataset_train)[i]
        
        group = dataset_train.groupby(key)['Income'].sum() / dataset_train.groupby(key)['Income'].size()
        group.plot(kind='bar')
        
print(dataset_train.shape)


# ## Data preprocessing

# In[47]:


# Encoding categorial data
# Replace string by categories number
drop = []
train_objs_num = len(train)

dataset = train.append(test)
print(train.shape)
print(test.shape)
print(dataset.shape)

# Salary Mapping

###############################################
# Mapping Method 1
##############################################

edu_mapping = {
        ' Preschool':0, 
        ' 1st-4th':1,
        ' 5th-6th':2,
        ' 7th-8th':3,
        ' 9th':4,
        ' 10th':5,
        ' 11th':6,
        ' 12th':7,
        ' HS-grad':8,
        ' Some-college':9,
        ' Assoc-voc':10,
        ' Assoc-acdm':11,
        ' Bachelors':12,
        ' Masters':13,
        ' Prof-school':14,
        ' Doctorate':15
}
gender_mapping = {
    ' Female':0,
    ' Male':1
}
race_mapping = {
    ' Amer-Indian-Eskimo':1,
    ' White':4, 
    ' Asian-Pac-Islander':3, 
    ' Other':0, 
    ' Black':2
}
marital_mapping={
    ' Married-civ-spouse':1,
    ' Divorced':0,
    ' Never-married':1,
    ' Separated':0,
    ' Widowed':0,
    ' Married-spouse-absent':1,
    ' Married-AF-spouse':1
}
relation_mapping={
    ' Wife':2,
    ' Husband':2,
    ' Own-child':1,
    ' Otherrelative':1,
    ' Not-in-family':0,
    ' Unmarried':0
}
#['Age','Workclass','fnlwgt','Education','Education-num','Marital-status','Occupation',' \
#                     Relationship','Race','Sex','Capital-gain','Capital-loss','Hours-per-week','Native-country','Income']

dataset['Education'] = dataset['Education'].map(edu_mapping)
dataset['Sex'] = dataset['Sex'].map(gender_mapping)
dataset['Race'] = dataset['Race'].map(race_mapping)
dataset['Marital-status'] = dataset['Marital-status'].map(marital_mapping)

dataset.loc[dataset['Native-country'] != ' United-States', 'Native-country'] = 'Non-US'
dataset.loc[dataset['Native-country'] == ' United-States', 'Native-country'] = 'US'
dataset['Native-country'] = dataset['Native-country'].map({'US':1,'Non-US':0}).astype(int)

income_minus = dataset.values[:,10] - dataset.values[:,11]
min_value = min(income_minus)
dataset['Capital-gain'] = (income_minus.astype(int))



###############################################
# Mapping Method 2
##############################################
'''
dataset.loc[dataset['Native-country'] != ' United-States', 'Native-country'] = 'Non-US'
dataset.loc[dataset['Native-country'] == ' United-States', 'Native-country'] = 'US'
dataset['Native-country'] = dataset['Native-country'].map({'US':1,'Non-US':0}).astype(int)

dataset['Marital-status'] = dataset['Marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
dataset['Marital-status'] = dataset['Marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
dataset['Marital-status'] = dataset['Marital-status'].map({'Couple':0,'Single':1})

def f(x):
    if x['Workclass'] == ' Federal-gov' or x['Workclass']== ' Local-gov' or x['Workclass']==' State-gov': return 'govt'
    elif x['Workclass'] == ' Private':return 'private'
    elif x['Workclass'] == ' Self-emp-inc' or x['Workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
dataset['Workclass']=dataset.apply(f, axis=1)  

dataset.loc[(dataset['Capital-gain'] > 0),'Capital-gain'] = 1
dataset.loc[(dataset['Capital-gain'] == 0 ,'Capital-gain')]= 0
dataset.loc[(dataset['Capital-loss'] > 0),'Capital-gain'] = 1
dataset.loc[(dataset['Capital-loss'] == 0 ,'Capital-gain')]= 0
'''

# One hot encoder
'''
for row in range(0, dataset.shape[1]):
    if (isinstance(dataset.values[1][row], str)):        
        print ("Delete row: ", row), 
        one_hot = pd.get_dummies(dataset[row])
        drop.append(row)
        dataset = pd.concat([dataset, one_hot], axis=1)
        dataset = dataset.iloc[:, :-1]
    
# Remove original attributes
drop.sort(reverse=True)
for row in drop:
    dataset = dataset.drop(row, axis = 1)
'''

# Only label encoder
for row in range(0, dataset.shape[1]):
    if (isinstance(dataset.values[1][row], str)): 
        print (row),
        labelencoder = LabelEncoder()
        target = labelencoder.fit_transform(dataset.values[:, row])
        key = list(dataset)[row]
        dataset[key] = target
#dataset = dataset.drop('Capital-loss', axis = 1)
print(dataset.shape)
print ("Finish One Hot Enconding")

for row in range(0, dataset.shape[1]):
    key = list(dataset)[row]
    dataset[key] = dataset[key].astype(int)
# PCA
'''
n_com = 30
pca = PCA(n_components=n_com)
dataset_pca = pca.fit_transform(dataset)
print ("Finish PCA preprocess") 
'''


# In[37]:


X = dataset[:train_objs_num]
X = pd.concat([X,dataset_train['Income']], axis = 1) 
print(X.shape)
X


# In[48]:


#correlation matrix
print(X.shape)
X = dataset[:train_objs_num]
X = pd.concat([X,dataset_train['Income']], axis = 1) 
X['Income'] = X['Income'].astype(int)

print(type(X['Age']), type(X['Age'][0]))
print(type(X['Education']), type(X['Education'][0]))
#print(X['Age'].corr(X['Education']))

corrmat = X.corr()
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[39]:


k = 15 #number of variables for heatmap
fig = plt.figure(figsize=(20,20))
cols = corrmat.nlargest(k, 'Income')['Income'].index
cm = np.corrcoef(X[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[52]:


#['Age','Workclass','fnlwgt','Education','Education-num','Marital-status','Occupation',' \
#                     Relationship','Race','Sex','Capital-gain','Capital-loss','Hours-per-week','Native-country','Income']
print(dataset.columns)
dataset_preprocess = dataset.copy()
dataset_preprocess = dataset_preprocess.drop('Education', axis = 1)
dataset_preprocess = dataset_preprocess.drop('fnlwgt', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Workclass', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Native-country', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Occupation', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Race', axis = 1)

###
#dataset_preprocess = dataset_preprocess.drop('Sex', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Capital-gain', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Relationship', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Age', axis = 1)
#dataset_preprocess = dataset_preprocess.drop('Hours-per-week', axis = 1)
print(dataset_preprocess.columns)
print(dataset_preprocess.shape)


# ## XGBOOST

# In[53]:


from sklearn.grid_search import GridSearchCV
import warnings
import itertools
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
'''
optimization_dict = {'max_depth': [4,5,6,7],
                     'n_estimators': [370],
                     'learning_rate':[0.1,0.8,0.12],
                     'subsample':[1.0,0.9],
                     'gamma':[0.2,0.3],
                     'colsample_bytree':[0.8]
                    }

xgb = XGBClassifier(tree_method = 'gpu_hist', predictor = 'gpu_predictor')
xgbc = GridSearchCV(xgb, optimization_dict, scoring='f1_micro', verbose=1, cv=3)
xgbc.fit(X_train, y_train)
print (xgbc.best_params_)

y_pred_max = xgbc.predict(X_test)
'''
n_set = [290, 370, 450]
depth = [4,5,6,7,8]
learn = [0.08,0.1,0.12]
gam = [0.2,0.3]
col = [0.7,0.8,1.0]
sub= [1.0,0.9]
f_s = 0
for n, d, l, g, c, s in itertools.product(n_set, depth, learn, gam, col, sub):
    xgbc = XGBClassifier(tree_method = 'gpu_hist', n_estimators=n, max_depth=d, learning_rate=l, cv=3, gamma=g, colsample_bytree=c, subsample=s)
    xgbc.fit(X_train, y_train)
    y_pred_max = xgbc.predict(X_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_max, average='micro')
    if(fscore > f_s):
        f_s = fscore
        print (n, d, l ,g, c, s, "fscore = ", f_s)

print ("Misclassified sample %d" % (y_test!=y_pred_max).sum())
print ("Train Accuracy: ", accuracy_score(y_train, xgbc.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred_max))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_max, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)


# In[54]:


X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
xgbc = XGBClassifier(n_estimators=290, max_depth=4, learning_rate=0.1, cv=3, gamma=0.2, colsample_bytree=0.8, subsample=0.9)
xgbc.fit(X_train, y_train)
y_pred_max = xgbc.predict(X_test)

print ("Misclassified sample %d" % (y_test!=y_pred_max).sum())
print ("Train Accuracy: ", accuracy_score(y_train, xgbc.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred_max))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_max, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)

test_std_pred = xgbc.predict(test_data)

with open('answer.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## SVM

# In[109]:


X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]
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
print ("Train Accuracy: ", accuracy_score(y_train, svm.predict(X_train_std)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)


test_std = sc.transform(test_data)
test_std_pred = svm.predict(test_std)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## Random Forest Tree

# In[141]:


X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
max_fscore = 0
target_n_estimators = 0
for i in range(10, 100):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=i)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
    if fscore > max_fscore:
        max_fscore = fscore
        target_n_estimators = i
'''    
forest = RandomForestClassifier(criterion='entropy', n_estimators=20)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, forest.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)


test_std = sc.transform(test_data)
test_std_pred = forest.predict(test_std)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## Logistic Regression

# In[186]:


X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
logreg = LogisticRegression(C=1.01, solver='lbfgs', multi_class='multinomial', penalty = 'l2')
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

logreg.fit(X_train_std, y_train)
y_pred = logreg.predict(X_test_std)
print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, logreg.predict(X_train_std)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)

test_std = sc.transform(test_data)
test_std_pred = logreg.predict(test_std)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## Bagging

# In[217]:


X = dataset_preprocess[:train_objs_num]
test_data = dataset_preprocess[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

bag = BaggingClassifier(n_estimators = 20)
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)

#scores = cross_val_score(xgbc, X_train, y_train, cv=10, scoring='accuracy')
#print (scores)

print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, bag.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)


test_std_pred = bag.predict(test_data)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## MLP

# In[126]:


X = dataset_preprocess.iloc[:train_objs_num]
test_data = dataset_preprocess.iloc[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(300, 300), max_iter=100, alpha=1e-5,
                    solver='sgd', verbose=1, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)

clf.fit(X_train_std, y_train) 

y_pred = clf.predict(X_test_std)

#scores = cross_val_score(xgbc, X_train, y_train, cv=10, scoring='accuracy')
#print (scores)

print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, clf.predict(X_train_std)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)

test_std = sc.transform(test_data)
test_std_pred = clf.predict(test_std)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## Gradient Boost

# In[166]:


X = dataset_preprocess.iloc[:train_objs_num]
test_data = dataset_preprocess.iloc[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


gbc = GradientBoostingClassifier(n_estimators=200, learning_rate = 0.1, max_depth = 25, random_state = 0)

gbc.fit(X_train, y_train) 

y_pred = gbc.predict(X_test)

#scores = cross_val_score(xgbc, X_train, y_train, cv=10, scoring='accuracy')
#print (scores)

print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, gbc.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)


test_std_pred = gbc.predict(test_data)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")


# ## Label propagation

# In[85]:


from sklearn.semi_supervised import label_propagation, LabelPropagation

X = dataset_preprocess.iloc[:train_objs_num]
test_data = dataset_preprocess.iloc[train_objs_num:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


label_spread = label_propagation.LabelSpreading()

label_spread.fit(X_train_std, y_train) 

y_pred = label_spread.predict(X_test_std)

#scores = cross_val_score(xgbc, X_train, y_train, cv=10, scoring='accuracy')
#print (scores)

print ("Misclassified sample %d" % (y_test!=y_pred).sum())
print ("Train Accuracy: ", accuracy_score(y_train, label_spread.predict(X_train_std)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
print ("precision: ", precision)
print ("recall: ", recall)
print ("fscore: ", fscore)

test_std = sc.transform(test_data)
test_std_pred = label_spread.predict(test_std)

with open('output.csv', 'w+') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'ans'])
    for i in range(len(test_std_pred)):
        writer.writerow([i, test_std_pred[i]])

print ("Done")

