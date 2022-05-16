#NN using scikit

from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas
from sklearn.model_selection import train_test_split

df=pandas.read_csv('train_set1.csv')
df1 = df
df1 = df1[~df1.group.str.contains("NE")]

df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]
df1['avg_spikes_burst'] = df1['avg_spikes_burst'].fillna(0)

df1=df1.drop(columns=['animal','bf', 'bfr', 'avg_ibi'])
df1=df1.dropna()
y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['group'])

X = StandardScaler().fit_transform(df1.values)

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


#NN model
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

#fit the model
#clf = MLPClassifier(solver = 'lbfgs', random_state=0, max_iter=5000).fit(X_train, y_train)


# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

cv_outer = KFold(n_splits = 10, random_state= 1, shuffle=True)
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    
    cv_inner =  KFold(n_splits = 10, random_state= 1, shuffle=True)
    model = MLPClassifier(random_state=0, max_iter=5000, solver = 'lbfgs')#.fit(X_train, y_train)
    space = dict()
    space['alpha'] = [0.0001, 0.001, 0.01]
    space['epsilon'] = [0.00000001, 0.001, 0.01]
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    result = search.fit(X_train, y_train)
    best_model = result.best_estimator_
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    
    
    
    
    
#%%
# =============================================================================
# from sklearn.model_selection import RepeatedKFold
# 
# cvr = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# model =  MLPClassifier(random_state=0, max_iter=5000)
# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cvr, n_jobs=-1)
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# 
# =============================================================================
