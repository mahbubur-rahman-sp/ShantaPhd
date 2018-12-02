import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1, max_iter=40)
feat_selector.fit(X.as_matrix(), y.as_matrix())

X_new = feat_selector.transform(X.as_matrix())


# rfe = RFE(estimator= svc, n_features_to_select= 50, step= 3).fit(X,y)
# X_new = rfe.transform(X)

print(X_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20)  


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# #forest = RandomForestClassifier(n_jobs=-1)

#svclassifier = SVC(kernel='sigmoid')X_train

rf_random.fit(X_train, y_train)  
forest = rf_random.best_estimator_

y_pred = forest.predict(X_test)  
print(forest.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  