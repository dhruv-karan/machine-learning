# K-Means Clustering

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

# finding optimal numbe rof cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)


#applying kmeans on datet
Kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans =Kmeans.fit_predict(X)

#vsiualise the Cluster
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='black',label='sensible')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('cluster of clients')
plt.xlabel('annnual oncole')
plt.ylabel('spending score')
plt.legend()
plt.show()