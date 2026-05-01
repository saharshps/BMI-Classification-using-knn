import pandas as pd 
import numpy as np

df=pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\knn.csv")
print(df)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print(df.columns)

df['Class']=le.fit_transform(df['Class'])
print(df)

from sklearn.model_selection import train_test_split
X=df.drop('Class',axis=1)
y=df['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))
print(knn.score(X_train,y_train))

y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_test, y_pred, average='weighted'))
print(recall_score(y_test, y_pred, average='weighted'))
print(f1_score(y_test, y_pred, average='weighted'))

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(knn, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

from sklearn.model_selection import StratifiedKFold,ShuffleSplit,KFold
skf=StratifiedKFold(n_splits=5,
                    shuffle=True,
                    random_state=42)

cv_scores_skf=cross_val_score(knn,X,y,cv=skf)

print("StratifiedKFold cross-validation scores:", cv_scores_skf)
print("Average StratifiedKFold cross-validation score:", np.mean(cv_scores_skf))

ss=ShuffleSplit(n_splits=5,
                test_size=0.2,
                random_state=42)

cv_scores_ss=cross_val_score(knn,X,y,cv=ss)

print("ShuffleSplit cross-validation scores:", cv_scores_ss)
print("Average ShuffleSplit cross-validation score:", np.mean(cv_scores_ss))

kf=KFold(n_splits=5,
         shuffle=True,
         random_state=42)
cv_scores_kf=cross_val_score(knn,X,y,cv=kf)

print("KFold cross-validation scores:", cv_scores_kf)
print("Average KFold cross-validation score:", np.mean(cv_scores_kf))

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
grid_search = GridSearchCV(estimator=knn, 
                           param_grid=param_grid, 
                           cv=5)

grid_search.fit(X, y)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2])

y_prob = knn.predict_proba(X_test)

roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr')

print("AUC:", roc_auc)

import joblib
joblib.dump(knn, 'knn_model.pkl')
