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
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
from keras.utils import multi_gpu_model

# Import `Dense` from `keras.layers`
from keras.layers import Dense

bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 




forest =RandomForestClassifier(n_jobs=-1)

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X.as_matrix(), y.as_matrix())

X = feat_selector.transform(X.as_matrix())

print(X.shape[1])

clusters = 8

model = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto', n_clusters=clusters,
    random_state=None, tol=0.0001, verbose=0)
model.fit(X)
alldistances = model.fit_transform(X)

np.append(X,alldistances,axis=1)

# for i in range(clusters):
#     X['d'+str(i)] = pd.Series(alldistances[:,i])
   

#forest = RandomForestClassifier(n_jobs=-1)



# rfe = RFE(estimator= svc, n_features_to_select= 50, step= 3).fit(X,y)
# X_new = rfe.transform(X)
#X_new = X

print(X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

with tf.device('/cpu:0'):
    model = Sequential()

    model.add(Dense(1024, activation='relu', input_shape=(X.shape[1],)))


    # Add one hidden layer 
    model.add(Dense(1024, activation='relu'))
    # Add one hidden layer 
    model.add(Dense(1024, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, activation='sigmoid'))

    # corr = wines.corr()
    # sns.set()
    # sns.heatmap(corr, 
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    # plt.show()

    # Model output shape
    model.output_shape

    # Model summary
    model.summary()

    # Model config
    model.get_config()

# List all weight tensors 
    model.get_weights()
parallel_model = multi_gpu_model(model, gpus=8)

parallel_model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
                
parallel_model.fit(X_train, y_train,epochs=100, batch_size=32*8, verbose=1)

y_pred = parallel_model.predict(X_test)

score = parallel_model.evaluate(X_test, y_test,verbose=1)

print(score)
# print(min(labels) )   
# print(max(labels) )   
# print(confusion_matrix(y,labels))
# print(classification_report(y,labels))  