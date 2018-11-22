import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  


bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 

#X_new = SelectKBest(chi2, k=90).fit_transform(X, y)

#lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(X, y)

#clf = ExtraTreesClassifier(n_estimators=50).fit(X, y)

svc = SVC(kernel="linear", C=1)

#model = SelectFromModel(svc, prefit=True)
#X_new = model.transform(X)


rfe = RFE(estimator= svc, n_features_to_select= 50, step= 3).fit(X,y)
X_new = rfe.transform(X)

print(X_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20)  



svclassifier = SVC(kernel='linear')  
#svclassifier = SVC(kernel='sigmoid')X_train

svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  