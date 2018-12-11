import pandas as pd  
import numpy as np  
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from PseudoLabeler import PseudoLabeler 
from boruta import BorutaPy
# import http.client



target = 'IS_MALWARE'

labelData = shuffle(pd.read_csv("label.csv"))
X_label = labelData.drop('IS_MALWARE', axis=1)  
y_label = labelData['IS_MALWARE'] 


unLabelData = shuffle(pd.read_csv("Unlabel.csv"))
X_test = unLabelData.drop('IS_MALWARE', axis=1)  
y_test = unLabelData['IS_MALWARE'] 

features = X_test.columns[0:]
print(features.shape)

feat_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1,max_iter=20)
feat_selector.fit(X_label.as_matrix(), y_label.as_matrix())
X_label = feat_selector.transform(X_label.as_matrix())


model_factory = [
    ExtraTreesClassifier(n_estimators=100),
    PseudoLabeler(
    ExtraTreesClassifier(n_estimators=100),
        X_test,
        features,
        target,
        sample_rate=0.25
    ),

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
    model.seed = 42
    kf = KFold(n_splits=10)
   


    scores = cross_val_score(model, X_label, y_label, cv=kf)
    #scores = cross_val_score(model, X_label, y_label, cv=num_folds, scoring='neg_mean_squared_error')
    score_description = " %0.4f" % scores.mean()

    print('{model:25} CV-5 RMSE: {score}'.format(
    model=model.__class__.__name__,
    score=score_description
    ))