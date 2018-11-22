import pandas as pd  
import numpy as np  
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 

model = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
model.fit(X)

k_means_labels = model.labels_
k_means_cluster_centers = model.cluster_centers_
dists = euclidean_distances(k_means_cluster_centers)

print(dists)


labels = model.predict(X)
for i in range(len(labels)):
    labels[i] =  labels[i] 


print(min(labels) )   
print(max(labels) )   
print(confusion_matrix(y,labels))
print(classification_report(y,labels))  