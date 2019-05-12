import pandas as pd
import numpy as np  
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
from sklearn.cluster import KMeans
import math
import random
from PseudoLabeler import PseudoLabeler 

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
    return model.score(X_test, y_test)


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
X_label = X_All[0:labelDataCount]
X_test = X_All[labelDataCount:]
features = X_test.columns[0:]




model_factory = [
     ExtraTreesClassifier(n_estimators=100),
    # PseudoLabeler(
    # ExtraTreesClassifier(n_estimators=100),
    #     X_test,
    #     features,
    #     target,
    #     sample_rate=0.25
    # ),

 #RandomForestClassifier(n_estimators=600,bootstrap=False,max_depth=60,max_features='sqrt',min_samples_split=5,min_samples_leaf=1)
#  XGBClassifier(nthread=1),
#  MLPRegressor(),
#  ExtraTreesClassifier(),
#  KNeighborsClassifier(),
#  GradientBoostingClassifier(),
#  SVC(kernel="linear", C=1),
#  SVC(kernel='sigmoid')
 
]




for model in model_factory:
    print(calculateAccuracy(model,X_label,y_label))

