import pandas as pd
import numpy as np  
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics 
from sklearn.cluster import KMeans
import math
import random
from PseudoLabeler import PseudoLabeler 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from boruta import BorutaPy

# Import `Dense` from `keras.layers`
tf = []
idf = []

def calculateCrossFoldAccuracy(model,data,target):
    model.seed = random.randint(41,91)
    kf = KFold(n_splits=10)
    scores = cross_val_score(model, data, target, cv=kf)
    score_description = " %0.4f" % scores.mean()

    print('{model:25} CV-5 RMSE: {score}'.format(
    model=model.__class__.__name__,
    score=score_description
    ))



def calculateAccuracy(model,data,target):
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.20)  
    model.fit(X_train, y_train)  
    scor = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred, average='macro') 
    precision_score = metrics.precision_score(y_test, y_pred, average='macro')
    
    
    y_pred_proba = model.predict_proba(X_test)[::,-1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    #plt.show()

    cm = metrics.confusion_matrix(y_test, y_pred) 
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)

    print('{tpr} {fpr} {Precision} {AUC} {score}'.format(
    tpr=TPR, fpr=FPR, Precision = precision_score,AUC = auc,
    score=scor
    ))

 


def calculateTF(row):
    global tf
    sum = row.sum()
    if sum == 0:
        sum = 1
    tfRow = [1.0*x/sum for x in row]
    tf.append(tfRow)

def calculateIDF(col):
    global idf
    existIn = np.count_nonzero(col)
    idf.append(math.log10(len(col)/existIn) if existIn>0 else 0)

def calculateTfIdf(tf,idf):
    for row in tf:
        for i in range(10):
            row[i] *= idf[i]
    return tf        


labelData = shuffle(pd.read_csv("dangerousPerm1.csv"))
X_unlabel = shuffle(pd.read_csv("dangerousPermUnlabel.csv"))
X_label = labelData.drop('IS_MALWARE', axis=1)  
y_label = labelData['IS_MALWARE'] 
target = 'IS_MALWARE'
X =  pd.concat([X_label, X_unlabel], ignore_index=True)

frequencies = X[X.columns[-10:]]
frequencies.apply(calculateTF,axis=1)
frequencies.apply(calculateIDF)
tfidf = calculateTfIdf(tf,idf)

cols = [x+'_tf_idf' for x in X_label.columns[-10:]]  
tfidf= pd.DataFrame(tfidf, columns=cols)

# tfidf = pd.read_csv("out.csv")
labelDataCount = len(X_label.index)


kmn = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_init=10, n_jobs=1, precompute_distances='auto', n_clusters=3,
    random_state=None, tol=0.0001, verbose=0)

alldistances = kmn.fit_transform(tfidf)
alldistances_label = alldistances[0:labelDataCount]
# model =RandomForestClassifier(n_jobs=-1, n_estimators=100)
# (calculateCrossFoldAccuracy(model,X_label,y_label))
# model = ExtraTreesClassifier(n_estimators=100)
# (calculateCrossFoldAccuracy(model,X_label,y_label))


#X_label = pd.concat([X_label, tfidfLabel],axis=1)

X_distance= pd.DataFrame(alldistances, columns=['dist1','dist2','dist3'])

X_All = pd.concat([X,X_distance],axis=1)
x_label = X_All[0:labelDataCount].values
X_unlabel = X_All[labelDataCount:].values





# labelData = shuffle(pd.read_csv("dangerousPerm1.csv"))
# X_unlabel = shuffle(pd.read_csv("dangerousPermUnlabel.csv"))

features = labelData.columns[0:-1]
# x_label = labelData[features].values
# y_label = labelData['IS_MALWARE'].values 
# X_unlabel = X_unlabel[features].values

print(x_label.shape)

lin_clf= RandomForestClassifier(n_estimators=100)  

# feat_selector = BorutaPy(lin_clf, n_estimators='auto', verbose=2, random_state=1,max_iter=20)
# feat_selector.fit(x_label, y_label)
# x_label = feat_selector.transform(x_label)
# X_unlabel = feat_selector.transform(X_unlabel)

print(x_label.shape)
#lin_clf = XGBClassifier(nthread=1)
#lin_clf= SVC(kernel="linear", C=1,probability=True)
calculateAccuracy(lin_clf,x_label,y_label)  
lin_clf.fit(x_label,y_label) 





  


for i in range(20): 
    probability = lin_clf.predict_proba(X_unlabel)
    pseudoY_test = lin_clf.predict(X_unlabel)

    X_taken =[]
    Y_taken =[]
    X_Rest = []
    for i in range(len(pseudoY_test)):
        prob = probability[i]
        if math.fabs(prob[1]-prob[0]) > 0.5:
            X_taken.append(X_unlabel[i]) 
            Y_taken.append(pseudoY_test[i])  
        else:
            X_Rest.append(X_unlabel[i])    

    print(len(X_taken))    
    X_unlabel = X_Rest

    x_label = np.vstack((x_label, X_taken))
    y_label = np.concatenate((y_label, Y_taken), axis=0)
    lin_clf= RandomForestClassifier(n_estimators=100) 
    #lin_clf = XGBClassifier(nthread=1)  
    #lin_clf= SVC(kernel="linear", C=1, probability=True)
    calculateAccuracy(lin_clf,x_label,y_label)      
    lin_clf.fit(x_label,y_label)
    if len(X_Rest)<50:
        break