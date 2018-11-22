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

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Read in white wine data 
wines = pd.read_csv("perm_call_feature.csv", sep=',')

X=wines.ix[:,0:500]

# Specify the target labels and flatten the array
y= np.ravel(wines.IS_MALWARE)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

with tf.device('/cpu:0'):
    model = Sequential()

    model.add(Dense(1024, activation='relu', input_shape=(500,)))


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