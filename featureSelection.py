import pandas as pd  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


bankdata = pd.read_csv("perm_call_feature.csv")

X = bankdata.drop('IS_MALWARE', axis=1)  
y = bankdata['IS_MALWARE'] 


print(X.shape)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new[1:])