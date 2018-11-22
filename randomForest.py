import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  
from boruta import BorutaPy


bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 

#X_new = SelectKBest(chi2, k=90).fit_transform(X, y)

#lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X, y)

#clf = ExtraTreesClassifier(n_estimators=50).fit(X, y)

forest = RandomForestClassifier(n_jobs=-1)
#forest = ExtraTreesClassifier(n_estimators=50)

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
feat_selector.fit(X.as_matrix(), y.as_matrix())

X_new = feat_selector.transform(X.as_matrix())


# rfe = RFE(estimator= svc, n_features_to_select= 50, step= 3).fit(X,y)
# X_new = rfe.transform(X)

print(X_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20)  




# #forest = RandomForestClassifier(n_jobs=-1)
svclassifier = SVC(kernel='sigmoid')  
#svclassifier = SVC(kernel='sigmoid')X_train

svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  