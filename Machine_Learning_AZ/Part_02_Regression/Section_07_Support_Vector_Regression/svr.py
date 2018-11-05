import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
# SVR lib doesn't have feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1) )

 
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Prediction a new result, we don't need fit here
y_pred = regressor.predict(sc_X.transform( np.array([[6.5]]) ))
y_pred = sc_y.inverse_transform(y_pred)
X_inverse = sc_X.inverse_transform(X)
y_inverse = sc_y.inverse_transform(y)

# Visualising the SVR results
plt.scatter(X_inverse, y_inverse, color = 'red')
plt.scatter(6.5, y_pred, color = 'green')
plt.plot(X_inverse, sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()