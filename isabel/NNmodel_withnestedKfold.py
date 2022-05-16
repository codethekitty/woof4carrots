#NN using scikit
#Use nested K Fold validation to determine good parameters

#imports for data selection
from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas

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



#NN model imports
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


#kfold imports
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#initialize KFold and outer results
cv_outer = KFold(n_splits = 3, random_state= 1, shuffle=True)
outer_results = list()
#For loop to run scan 
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    #define cross validation number of folds - should be 3 or 5
    cv_inner =  KFold(n_splits = 2, random_state= 1, shuffle=True)
    #specify 
    model = MLPClassifier(random_state=0, max_iter=5000, solver = 'adam')#.fit(X_train, y_train)
    space = dict()
    #trying different alpha values - regularization term
    space['alpha'] = [0.0001, 0.001, 0.01]
    #trying different epsilon - only valid for adam solver - value for numerical stability
# =============================================================================
#     space['epsilon'] = [0.00000001, 0.001, 0.01]
#     #trying different beta_1 values - exponential decay rate for 1st moment vector
#     space['beta_1'] = [0.85, 0.9, 0.92]
#     #trying different beta_2 values - exponential decay rate for 2nd moment vector
#     space['beta_2'] = [0.99, 0.995, 0.999]
# =============================================================================
    
    #searching parameters and refitting
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    result = search.fit(X_train, y_train)
    
    #returns best refit param
    best_model = result.best_estimator_
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    


#see this tutorial: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/