# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


np.random.seed(0)

X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)

y_xor = np.where(y_xor,1,-1)

# =================================visualling the data set

plt.scatter(X_xor[y_xor == 1,0],X_xor[y_xor ==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor ==-1,0],X_xor[y_xor == -1,1],c='r',marker='s',label ='-1')

plt.ylim(-3.0)
plt.legend()
plt.show()


def plot_decision_region(X,y,classifier,resolution=0.02):
    #setup marker 
    colors = ('red','blue')
    cmap = ListedColormap(colors[:2])
    #plot the decision surfCE
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
 
    


# separting it 

from sklearn.svm import SVC

svm = SVC(kernel='rbf',random_state=0,gamma=0.10, C=10.0)
svm.fit(X_xor,y_xor)
plot_decision_region(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()





