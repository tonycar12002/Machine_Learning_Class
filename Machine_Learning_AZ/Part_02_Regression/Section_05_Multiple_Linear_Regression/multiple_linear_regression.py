# y = bo + b1*x1 + b2*x2 + b3*x3 ......

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# Take all columns except last one
X = dataset.iloc[:, :-1].values
# Take only column3
y = dataset.iloc[:, 4].values


# Create dummy variables
# Becareful about dummy variables trap 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoer_X = LabelEncoder()
X[:, 3] = labelencoer_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
# remove first column of X
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regresiion to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# insert columns with x0 = 1 into first column
# y = bo + b1*x1 + b2*x2 + b3*x3  => y = bo*x0 + b1*x1 + b2*x2 + b3*x3
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
regressor_OLS = LinearRegression()

def backwardElimination(x, sl):
    results = sm.OLS(endog = y, exog= x).fit()
    max_p_index = np.argmax(results.pvalues)
    max_p = results.pvalues[max_p_index]
    
    if max_p >= sl:
        x = np.delete(x, max_p_index, axis =1)
        backwardElimination(x, sl)
    else:
        print (results.summary())
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
results = backwardElimination(X_opt, 0.08)



















