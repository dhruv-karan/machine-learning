# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#ploynomical reg
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualling linear regression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('salry vs position')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
#visiualising poly 
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('salary vs postions')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
#preicting iin linear
lin_reg.predict(6.5)
#predidcting poly

lin_reg_2.predict(poly_reg.fit_transform(6.5))