# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[: ,[2,4]].values
y = dataset.iloc[:, 4].values

#seprating data set i n 2 parts 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=0 )

#performing feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting dataset to model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predicting
y_pred = classifier.predict(X_test)

## making a confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)