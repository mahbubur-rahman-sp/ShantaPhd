import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from boruta import BorutaPy


bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 

clusters = 8

model = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto', n_clusters=clusters,
    random_state=None, tol=0.0001, verbose=0)
model.fit(X)
alldistances = model.fit_transform(X)

for i in range(clusters):
    X['d'+str(i)] = pd.Series(alldistances[:,i])
   

#forest = RandomForestClassifier(n_jobs=-1)
forest = ExtraTreesClassifier(n_estimators=50)

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X.as_matrix(), y.as_matrix())

X_new = feat_selector.transform(X.as_matrix())


# rfe = RFE(estimator= svc, n_features_to_select= 50, step= 3).fit(X,y)
# X_new = rfe.transform(X)
#X_new = X

print(X_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20)  



forest.fit(X_train, y_train)  

y_pred = forest.predict(X_test)  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

# labels = model.predict(X)
# for i in range(len(labels)):
#     labels[i] =  labels[i] 


# print(min(labels) )   
# print(max(labels) )   
# print(confusion_matrix(y,labels))
# print(classification_report(y,labels))  