
#trim data above 120ms in training data

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

#Getting variables ready to read in files
Data = []
index = []
readin_Xtesting = [] 
readin_times = []
readin_unit = []
readin_testnames = []
#reading in all files ending in .npy (should be only relevant animals)
for f in npyFiles:
    #the files that are used for training and validating the data
    if 'training.npy' in f:
        s,syn,bur,meta=numpy.load(f,allow_pickle=True) # ignore syn,bur,meta
        Data.append(s)
        index.append(f[0:2])
    #the files that are used in the new testing of the data
    if 'spiketrain.npy' in f:
        
        s=numpy.load(f,allow_pickle=True)
        TestData = []
        t = []  
        testnamestemp = []

        for x in s:
            TestData.append(x['spiketrain'])
            t.append(x['t']) 
            readin_unit.append(x['unit'])
            testnamestemp.append(f[0:6])
        readin_Xtesting.append(TestData)
        readin_times.append(t)
        readin_testnames.append(testnamestemp[0])

y = []
X = []
key = []

#define targets as a list, to use in 'for' loop
animalNames = targets.animal.tolist()
#convert class_ in targets dataframe to list of numbers: ET = 1; ENT = 0
animalClassification = unique(targets.loc[:,'class_'].values,return_inverse=True)[1].tolist()

   
y = []
X = []
key = []

#define targets as a list, to use in 'for' loop
animalNames = targets.animal.tolist()
#convert class_ in targets dataframe to list of numbers: ET = 1; ENT = 0
animalClassification = unique(targets.loc[:,'class_'].values,return_inverse=True)[1].tolist()

#create X and y datasets based on 
for k, j in zip(animalNames, animalClassification):
    if k in index:      
        idx = [i for i, x in enumerate(index) if x == k]
        for i in idx:
            #i = animalNames.index(x)
            #j = index.index(x)
            y.append(j)
            X.append(Data[i])
            key.append(k)    

#%%
#Divide into 16 channels
X16 = []
y16 = []

nameKey = []

for x in range(0, len(X)):
    k = len(X[x])
    #print(x,y)
    if k == 16:
            X16.append(X[x])
            y16.append(y[x])
            nameKey.append(key[x])
    if k == 32:
            i = X[x]
            X16.append(i[0:16])
            X16.append(i[16:32])
            y16.append(y[x])
            y16.append(y[x])
            nameKey.append(key[x])
            nameKey.append(key[x])

#%%


bins = arange(0, 120.1, 0.1) #binned to 0.1 second
channels = []
for x in X16:
    chan = []
    for i in x:

        hist = np.histogram(i[ i <= 120],bins)
        chan.append(hist[0])
        
    channels.append(chan)
#%%

#Turns out we need all the numpy arrays to be the same length to fit the NN
X16 = []
y16 = []
X32 = []
y32 = []
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
            i = X[x]
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
X = []
for x in channels:
    df = pd.DataFrame(x)
    X.append(df.to_numpy().flatten())

from sklearn.preprocessing import StandardScaler    
X = pd.DataFrame(X) 
X = StandardScaler().fit_transform(X)
y = np.array(y16)

#Running the  model
from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0, max_iter=5000, solver = 'lbfgs')
mlp.fit(X_train, y_train)

from sklearn.metrics import plot_confusion_matrix
plt.figure()
plot_confusion_matrix(mlp, X_test, y_test) 
print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

#plot first x weights
x = 5
import matplotlib.pyplot as plt
fig, axes = plt.subplots(x, 1)
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(16,1200), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
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
from sklearn.metrics import plot_confusion_matrix

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
    model = MLPClassifier(random_state=0, max_iter=5000, solver = 'lbfgs')#.fit(X_train, y_train)
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
    plot_confusion_matrix(best_model, X_test, y_test) 
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    
plt.figure()
plot_confusion_matrix(best_model, X, y) 
parameters = best_model.get_params()
#see this tutorial: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/



#%%

##############################################################

########################################################

##################################################




#processing the new data
#get rid of times less than 7200
TestData_copy = TestData
times = []
time_list = []
for t in range(len(TestData)):  
    if TestData[t]['t'] < 120:
        times.append(t)
    else: 
        time_list.append(TestData[t]['t'])
for t in times:
     TestData_copy[t] = 'nan'
 
TestData_new = [x for x in TestData_copy if x != 'nan']     

#%%
#idx = [i for i, x in enumerate(index) if x == k]
from sklearn.preprocessing import StandardScaler    
#unique_times = set(time_list)
ttimes = []
tchannels = []
j = TestData_new[0]['t']
X16 = []
times = []
animals = []
bins = arange(0, 120.1, 0.1)
for r in range(len(TestData_new)):
    k = TestData_new[r]['t']
    if k == j:        
        ttimes.append(TestData_new[r]['t'])
        tchannels.append(TestData_new[r]['ts'])
    else:

        Xhist = np.zeros([130,2])
        #chan_dict[k] = tchannels
        for i in range(len(tchannels)):
            Xhist[i,0] = (np.std(np.histogram(diff(tchannels[i]))[0]))
            Xhist[i,1] = i
        a = Xhist[Xhist[:,0].argsort()]
        b = flip(a)
        b = b[0:16, 0]
        X16_temp = []
        for ii in b: 
            ii = np.array(ii.astype(numpy.int64))
            #print(np.array(ii.astype(numpy.int64)))
        #print(x[np.array(k.astype(numpy.int64))])
            chan_temp = tchannels[np.array(ii.astype(numpy.int64))]
            chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)

            X16_temp.append(chan_temp[0])
        df = pd.DataFrame(X16_temp)
        X16.append(df.to_numpy().flatten())
        times.append(TestData_new[r-1]['t'])
        animals.append(TestData_new[r-1]['f'])
       # X = pd.DataFrame(X)
       # Xtesting1 = StandardScaler().fit_transform(X) 
        
            
        #X16.append(X16_temp)
        
#restart for next time point
        tchannels = []
        tchannels.append(TestData_new[r+1]['ts'])
        ttimes.append(TestData_new[r+1]['t'])

    #chan_dict = {k:tchannels}
    j = TestData_new[r]['t']












# =============================================================================
# for r in range(len(TestData_new)):
#     k = TestData_new[r]['t']
#     if k == j:        
#         ttimes.append(TestData_new[r]['t'])
#         tchannels.append(TestData_new[r]['ts'])
#     else:
# 
#         ttimes.append(TestData_new[r+1]['t'])
#         chan_dict[k] = tchannels
#         
#         
#         
#         
#         #restart for next time point
#         tchannels = []
#         tchannels.append(TestData_new[r+1]['ts']) 
# 
#     #chan_dict = {k:tchannels}
#     j = TestData_new[r]['t']
#     
# =============================================================================
#%%

# =============================================================================
# X16 = []
# for j in range(0,len(i_tData)):
#     x = i_tData[j]
# 
#     Xhist = np.zeros([130,2])
#     X16_temp = []
#     print(len(x))
#     for i in range(0,len(x)):
#         
#         Xhist[i,0] = (np.std(np.histogram(diff(x[i]))[0]))
#         Xhist[i,1] = i
#     a = Xhist[Xhist[:,0].argsort()]
#     b = flip(a)
#     b = b[0:16, 0]
#     for k in b: 
#         x = i_tData[j]
#         #print(x[np.array(k.astype(numpy.int64))])
#         X16_temp.append(x[np.array(k.astype(numpy.int64))])
#     X16.append(X16_temp)
#     #print(j)   
# =============================================================================
#%%
#bins = arange(0, 120.1, 0.1) #binned to 0.1 second     
# =============================================================================
# Testchannels = []
# for x in X16:
#     chan = []
#     for i in x:
#         hist = np.histogram(i,bins)
#         chan.append(hist[0])
#         
#     Testchannels.append(chan)
# =============================================================================


# =============================================================================
# X = []
# for x in Testchannels:
#     df = pd.DataFrame(x)
#     X.append(df.to_numpy().flatten())
# =============================================================================
#%%

#Xtest_adjacency = Xtest_adjacency.fillna(0) #fill NaN with 0s rather than removing
# =============================================================================
# X = pd.DataFrame(X16)
# Xtesting = StandardScaler().fit_transform(X)     
# =============================================================================

from sklearn.preprocessing import StandardScaler    
X = pd.DataFrame(X16)
#Xtest_adjacency = Xtest_adjacency.fillna(0) #fill NaN with 0s rather than removing
Xtesting1 = StandardScaler().fit_transform(X)     
Xtesting = [] 
for x in Xtesting1:
    Xtesting.append(x)
removeNaNs = pd.DataFrame(
    {'Animal': animals,
     'Times': times,
     'data': Xtesting,
     })
#%%
i_timing = []
i_name_key = []
i_newXtest = []

for x,y,z in zip(removeNaNs['Animal'],removeNaNs['Times'], removeNaNs['data']):

    if np.isnan(sum(z)):
        print(sum(z))

    else:
        i_timing.append(y)        
        i_name_key.append(x)
        i_newXtest.append(z)

i_newXtest= pd.DataFrame(i_newXtest)


Xtesting = i_newXtest
#%%
# =============================================================================
# for a in range(0, len(removeNaNs['data'])):
#     r = removeNaNs['data'][a]
#     sumr = np.sum(r)
#     if np.isnan(sumr):
#         print(a)
#         removeNaNs = removeNaNs.drop(removeNaNs.index[a])
# #%%        
# for a in removeNaNs['data']:
#         if np.isnan(a):
#             removeNaNs = removeNaNs.drop(a)
# =============================================================================
    

#%%

testpredictions = best_model.predict(Xtesting)
testprob1 = best_model.predict_proba(Xtesting)
testprob = []
for i in testprob1:
    testprob.append(i[1])

#loop ii
ii_grouped_time = []
ii_grouped_names = []
ii_grouped_data = []
ii_grouped_predictions = []
ii_grouped_probability = []

unique_name = set(i_name_key)

for k in unique_name:
    ii_temp_animal = []
    ii_temp_grouped_data = []
    ii_temp_grouped_time = []
    ii_temp_grouped_predictions = []
    ii_temp_grouped_probability = []

    idxs = [i for i, x in enumerate(i_name_key) if x == k]
    for i in idxs:
        ii_temp_grouped_data.append(Xtesting[i])
        ii_temp_grouped_predictions.append(testpredictions[i])
        ii_temp_animal.append(i_name_key[i])
        ii_temp_grouped_time.append(i_timing[i])
        ii_temp_grouped_probability.append(testprob[i])

    ii_grouped_names.append(ii_temp_animal)
    ii_grouped_data.append(ii_temp_grouped_data)
    ii_grouped_time.append(ii_temp_grouped_time)
    ii_grouped_predictions.append(ii_temp_grouped_predictions)
    ii_grouped_probability.append(ii_temp_grouped_probability)



#Make a dataframe with all of the data together
New_sorted_data = pd.DataFrame(
    {'Animal': ii_grouped_names,
     'Times': ii_grouped_time,
     'Predictions': ii_grouped_predictions,
     'Probability': ii_grouped_probability,
     })


#%%
import matplotlib.pyplot as plt

for a in range(0, len(New_sorted_data['Animal'])):
    Animal_temp = (New_sorted_data.loc[a])
    figure()
    datas = numpy.array(Animal_temp['Probability'])
    times = numpy.array(Animal_temp['Times'])

    plt.plot(times,datas,'bo')

    
    plt.xlabel('time(s)')
    plt.xlim(0,max(Animal_temp['Times'])+10)
    plt.ylim(-0.1,1.1)
    plt.ylabel('Probability')
    plt.title("ST Probability of tinnitus for animal " +  str(Animal_temp['Animal'][0]))
    rcParams['pdf.fonttype'] = 42
    savefig('plot_animal' + str(Animal_temp['Animal'][0]) +'.pdf')

