import pandas as pd  
import numpy as np  
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, KFold,cross_validate , train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import tree
from PseudoLabeler import PseudoLabeler 
from boruta import BorutaPy
import datetime
# import http.client
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score







_scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'ruc_auc_score' : make_scorer(roc_auc_score),
           'f1_score' : make_scorer(f1_score)}




target = 'IS_MALWARE'

labelData = shuffle(pd.read_csv("label.csv"))
X_label = labelData.drop('IS_MALWARE', axis=1)  
y_label = labelData['IS_MALWARE'] 


X_train, X_test, y_train, y_test = train_test_split(X_label, y_label, test_size = 0.20)


model_factory = [
    ExtraTreesClassifier(n_estimators=10),
    tree.DecisionTreeClassifier(),
    KNeighborsClassifier(),
    RandomForestClassifier(n_estimators=10),
    GradientBoostingClassifier(),
    SVC(kernel="linear", C=1)
 
]


for model in model_factory:
    a = datetime.datetime.now()
    y_pred = model.fit(X_train,y_train).predict(X_test)
    
    #scores = cross_val_score(model, X_label, y_label, cv=num_folds, scoring='neg_mean_squared_error')
    _accuracy = " %0.4f" % accuracy_score(y_test, y_pred)
    _precision = " %0.4f" % precision_score(y_test, y_pred)
    _recall = " %0.4f" % recall_score(y_test, y_pred)
    _ruc_auc_score = " %0.4f" % roc_auc_score(y_test,y_pred)
    _f1_score = " %0.4f" % f1_score(y_test, y_pred)
    b = datetime.datetime.now()
    c = b-a
    print('{model:25} accuracy: {accuracy} precision: {precision} recall:{recall} ruc_auc_score:{ruc_auc_score} f1_score:{f1_score} time:{time}'.format(
    model=model.__class__.__name__,
    accuracy=_accuracy, precision=_precision,recall=_recall, ruc_auc_score=_ruc_auc_score,f1_score=_f1_score, time= c.seconds
    ))



# for model in model_factory:
#     model.seed = 42
#     kf = KFold(n_splits=10)
#     a = datetime.datetime.now()
#     scores = cross_validate (model, X_label, y_label, cv=kf,scoring=_scoring)
#     b = datetime.datetime.now()
#     c = b-a
#     #scores = cross_val_score(model, X_label, y_label, cv=num_folds, scoring='neg_mean_squared_error')
#     _accuracy = " %0.4f" % np.mean(scores['test_accuracy'])
#     _precision = " %0.4f" % np.mean(scores['test_precision'])
#     _recall = " %0.4f" % np.mean(scores['test_recall'])
#     _ruc_auc_score = " %0.4f" % np.mean(scores['test_ruc_auc_score'])
#     _f1_score = " %0.4f" % np.mean(scores['test_f1_score'])

#     print('{model:25} accuracy: {accuracy} precision: {precision} recall:{recall} ruc_auc_score:{ruc_auc_score} f1_score:{f1_score} time:{time}'.format(
#     model=model.__class__.__name__,
#     accuracy=_accuracy, precision=_precision,recall=_recall, ruc_auc_score=_ruc_auc_score,f1_score=_f1_score, time= c.seconds
#     ))