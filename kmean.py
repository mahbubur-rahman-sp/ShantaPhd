import pandas as pd  
import numpy as np  
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
from keras.utils import multi_gpu_model

# Import `Dense` from `keras.layers`
from keras.layers import Dense



def calculateAccuracy(model,data,target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.20)  
    model.fit(X_train, y_train)  
    return model.score(X_test, y_test)



labelData = shuffle(pd.read_csv("train.csv"))
X_label = labelData.drop('IS_MALWARE', axis=1)  
y_label = labelData['IS_MALWARE'] 


model =RandomForestClassifier(n_jobs=-1)
print(calculateAccuracy(model,X_label,y_label))


feat_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1)
feat_selector.fit(X_label.as_matrix(), y_label.as_matrix())
X_label = feat_selector.transform(X_label.as_matrix())

print(calculateAccuracy(model,X_label,y_label))



unlabelData = shuffle(pd.read_csv("test.csv"))

X_unlabel = unlabelData.drop('IS_MALWARE', axis=1)  

unlabelData1 = shuffle(pd.read_csv("test1.csv"))

X_unlabel1 = unlabelData1.drop('IS_MALWARE', axis=1)  


X_unlabel =  feat_selector.transform(X_unlabel.as_matrix())
X_unlabel1 =  feat_selector.transform(X_unlabel1.as_matrix())
print(X_label.shape)
X = np.concatenate((X_label,X_unlabel,X_unlabel1), axis=0)
print(X.shape)
kmn = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto', n_clusters=2,
    random_state=None, tol=0.0001, verbose=0)
kmn.fit(X)
alldistances = kmn.fit_transform(X_label)
print(X_label.shape)
X_label = np.append(X_label,alldistances,axis=1)
print(X_label.shape)
print(calculateAccuracy(model,X_label,y_label))



# forest =RandomForestClassifier(n_jobs=-1)

# feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
# feat_selector.fit(X_label.as_matrix(), y_label.as_matrix())

# X_label = feat_selector.transform(X_label.as_matrix())
# X = feat_selector.transform(X.as_matrix())

# print(X.shape[1])

# clusters = 2







#svclassifier = SVC(kernel='sigmoid')X_train

