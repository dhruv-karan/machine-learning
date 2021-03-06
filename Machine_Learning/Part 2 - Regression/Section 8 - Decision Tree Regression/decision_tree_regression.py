# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#fittig data set into regression t ree 
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#predictin the  values
y_pred = regressor.predict(6.5)

#visualting the regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('salary vs postion (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()