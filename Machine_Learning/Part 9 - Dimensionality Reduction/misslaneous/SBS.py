# -*- coding: utf-8 -*-

# ====== SBS ======== sequential backward propagation =====================
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.f_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self,X,y):
        X_train,X_test,y_train,y_test = \
                train_test_split(X,y,test_size=self.test_size,random_state = self.random_state)
        
        dim = X_train.shape[0]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._cal_score(X_train,y_train,X_test,y_test,self.indices_)
        
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores =[]
            subsets = []
            
            for p in  combinations(self.indices_,r=dim-1):
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -=1
            self.score_append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self,X):
        return X[:,self.indices_]
    
    def _cal_score(self,X_train,y_train,X_test,y_test,indices):
        self.estimator.fit(X_train[:,indices],y_train)
        y_pred =self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score
        
        
#==================== Varfying using KNN classifier
        
# -*- coding: utf-8 -*-


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
X = np.random.random((2000))
y = np.random.random((2000))
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X,y)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs._cal_scores_, marker='o')
plt.ylim([0.7,1.1])

plt.ylabel('Accuracy')
plt.xlabel('number of features')
plt.grid()
plt.show()


X.shape[1]
        
        
        