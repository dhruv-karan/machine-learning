# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
#performing feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#svr to data seet
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
#predicting 

y_predict = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualisation of svr
plt.scatter(X,y, color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.title('salary vs period')
plt.xlabel('period')
plt.ylabel('saly')
plt.show()