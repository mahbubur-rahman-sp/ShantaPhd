# Import pandas 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
# Import `Dense` from `keras.layers`
from keras.layers import Dense
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score, confusion_matrix
import datetime
# Read in white wine data 
target = 'IS_MALWARE'

labelData = shuffle(pd.read_csv("label.csv"))
X_label = labelData.drop('IS_MALWARE', axis=1)  

y_label = labelData['IS_MALWARE'] 



unlabeldata = shuffle(pd.read_csv("Unlabel.csv"))
X_unlabel = unlabeldata.drop('IS_MALWARE', axis=1)  

X = np.concatenate((X_label,X_unlabel), axis=0)

print(X.shape)
kmn = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto', n_clusters=2,
    random_state=None, tol=0.0001, verbose=0)
kmn.fit(X)
alldistances = kmn.fit_transform(X_label)
print(X_label.shape)
X_label = np.append(X_label,alldistances,axis=1)
print(X_label.shape)

X_train, X_test, y_train, y_test = train_test_split(X_label, y_label, test_size = 0.30)



scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

with tf.device('/cpu:0'):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu,input_shape=(502,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    # model = Sequential()

    # model.add(Dense(256, activation='relu', input_shape=(502,)))


    # # Add one hidden layer 
    # model.add(Dense(256, activation='relu'))
    # # Add one hidden layer 
    # model.add(Dense(256, activation='relu'))

    # model.add(Dense(256, activation='relu'))
    # model.dr
    # # Add an output layer 
    # model.add(Dense(1, activation='sigmoid'))

    # # corr = wines.corr()
    # sns.set()
    # sns.heatmap(corr, 
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    # plt.show()

    # Model output shape
    # model.output_shape

    # # Model summary
    # model.summary()

    # # Model config
    # model.get_config()

# List all weight tensors 
    # model.get_weights()
parallel_model = multi_gpu_model(model, gpus=8)

parallel_model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
a = datetime.datetime.now()                
parallel_model.fit(X_train, y_train,epochs=100, batch_size=32*8, verbose=1)

y_pred1 = parallel_model.predict(X_test)
y_pred = []
for item in y_pred1:
    y_pred.append(1 if item >= 0.5 else 0)

y_test1 = []
for item in y_test:
    y_test1.append(1 if item > 0.5 else 0)

_accuracy = " %0.4f" % precision_score(
    y_test1,
    y_pred
    
)
_precision = " %0.4f" % precision_score(y_test1, y_pred)
_recall = " %0.4f" % recall_score(y_test1, y_pred)
_ruc_auc_score = " %0.4f" % roc_auc_score(y_test1,y_pred)
_f1_score = " %0.4f" % f1_score(y_test1, y_pred)
b = datetime.datetime.now()
c = b-a

print(confusion_matrix(y_test1,y_pred))

print('{model:25} accuracy: {accuracy} precision: {precision} recall:{recall} ruc_auc_score:{ruc_auc_score} f1_score:{f1_score} time:{time} features:{features}'.format(
model='deep learning',
accuracy=_accuracy, precision=_precision,recall=_recall, ruc_auc_score=_ruc_auc_score,f1_score=_f1_score, time= c.seconds,features = 502))