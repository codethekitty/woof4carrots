import numpy
import glob
import pandas as pd
import numpy as np
from pylab import*

#reading in cvs of animals plus their class
targets = pd.read_csv('exp_animal_list.csv')

#eliminate the weird classes
#I renamed the column class to class_ because of python syntax
targets = targets[~targets.class_.str.contains("NE")]
targets = targets[~targets.class_.str.contains("ET_T")]
targets = targets[~targets.class_.str.contains("ET_A")]
targets = targets[~targets.class_.str.contains("ET_E")]


#creating a glob of all .npy files in active folder
npyFiles = glob.glob("*.npy")

#Getting variables ready to read in files
Data = []
index = []

#reading in all files ending in .npy (should be only relevant animals)
for f in npyFiles:
    s,syn,bur,meta=numpy.load(f,allow_pickle=True) # ignore syn,bur,meta
    Data.append(s)
    index.append(f[0:2])
    
y = []
X = []
key = []

#define targets as a list, to use in 'for' loop
animalNames = targets.animal.tolist()
#convert class_ in targets dataframe to list of numbers: ET = 1; ENT = 0
animalClassification = unique(targets.loc[:,'class_'].values,return_inverse=True)[1].tolist()

#create X and y datasets based on 
for x in index:
    if x in animalNames:            
            i = animalNames.index(x)
            j = index.index(x)
            y.append(animalClassification[i])
            X.append(Data[j])
            key.append(x)    

#Turns out we need all the numpy arrays to be the same length to fit the NN
X16 = []
y16 = []

nameKey = []

for x in range(0, len(X)):
    k = 0
    k = len(X[x])
    #print(x,y)
    if k == 16:
            X16.append(X[x])
            y16.append(y[x])
            nameKey.append(key[x])
    if k == 32:
            i = Data[x]
            X16.append(i[0:16])
            X16.append(i[16:32])
            y16.append(y[x])
            y16.append(y[x])
            nameKey.append(key[x])
            nameKey.append(key[x])

bins = arange(0, 120.1, 0.1)
channels = []
for x in X16:
    chan = []
    for i in x:
        hist = np.histogram(i,bins)
        chan.append(hist[0])
        
    channels.append(chan)


#import to matlab to create adjacency matrix
from scipy.io import savemat
savemat("Data16.mat", {'X16': channels})
savemat("y16.mat", {'y': y16})

#Create adjacency matrix in python also
A = []
for x in channels:
    testcorr = np.stack(x, axis=0)
    testcorr = np.rot90(testcorr)
    df = pd.DataFrame(testcorr)
    Amat = df.corr()
    Amat = Amat - diag(diag(Amat))
    A.append(Amat)
X_adjacency = []
for x in A:
    X_adjacency.append(x.to_numpy().flatten())

from sklearn.preprocessing import StandardScaler    
X_adjacency = pd.DataFrame(X_adjacency) 
X = StandardScaler().fit_transform(X_adjacency)
y = np.array(y16)




#Running the  model
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=100, alpha=1e-3, beta_1 = 0.85, beta_2 =0.99, epsilon = 0.001,
                    solver='adam', verbose=10, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)

from sklearn.metrics import plot_confusion_matrix
plt.figure()
plot_confusion_matrix(mlp, X_test, y_test) 

print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

#%%

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
cv_outer = KFold(n_splits = 10, random_state= 1, shuffle=True)
outer_results = list()
#For loop to run scan 
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    #define cross validation number of folds - should be 3 or 5
    cv_inner =  KFold(n_splits = 3, random_state= 1, shuffle=True)
    #specify 
    model = MLPClassifier(random_state=0, max_iter=5000, solver = 'adam')#.fit(X_train, y_train)
    space = dict()
    #trying different alpha values - regularization term
    space['alpha'] = [0.0001, 0.001, 0.01]
    #trying different epsilon - only valid for adam solver - value for numerical stability
    space['epsilon'] = [0.00000001, 0.0001, 0.001]
    #trying different beta_1 values - exponential decay rate for 1st moment vector
    space['beta_1'] = [0.85, 0.9, 0.92]
    #trying different beta_2 values - exponential decay rate for 2nd moment vector
    space['beta_2'] = [0.99, 0.995, 0.999]
    
    #searching parameters and refitting
    search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
    result = search.fit(X_train, y_train)
    
    #returns best refit param
    best_model = result.best_estimator_
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    outer_results.append(acc)
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    plot_confusion_matrix(best_model, X_test, y_test) 
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    
plt.figure()
plot_confusion_matrix(best_model, X, y) 
parameters = best_model.get_params()
print("parameters")
print(parameters)
#see this tutorial: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/




