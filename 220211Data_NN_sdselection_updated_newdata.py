
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
targets = targets[~targets.class_.str.contains("NE")] #comment this out if u want to include these NE
targets = targets[~targets.class_.str.contains("ET_T")]
targets = targets[~targets.class_.str.contains("ET_A")]
targets = targets[~targets.class_.str.contains("ET_E")]

#targets.class_[targets.class_.str.contains("NE")] = "ENT" #include NE as ENT (as 0) ##leave commented out for multiclasss
#creating a glob of all .npy files in active folder
npyFiles = glob.glob("*.npy")


#Getting variables ready to read in files
TestData = []
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
        js=numpy.load(f,allow_pickle=True)
        for x in js:
            TestData.append(x)

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

import matplotlib.pyplot as plt
bins = arange(0, 120.1, 0.1) #binned to 0.001 second

for x in range(0, len(X)):
    k = len(X[x])
    #print(x,y)
    if k == 16:
            i = X[x]
            Xhist = np.zeros([130,2])
            for j in range(len(i)):
# =============================================================================
#                 figure()
#                 plt.hist(diff(i[j]))
# =============================================================================
                Xhist[j,0] = (np.std(np.histogram(diff(i[j]))[0]))
                Xhist[j,1] = j
            a = Xhist[Xhist[:,0].argsort()]
            b = flip(a)
            b = b[0:16, 0]
            X16_temp = []
            for ii in b: 
                ii = np.array(ii.astype(numpy.int64))
                chan_temp = i[np.array(ii.astype(numpy.int64))]
                chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)
                X16_temp.append(chan_temp[0])           
            X16.append(X16_temp)            
            y16.append(y[x])
            nameKey.append(key[x])

    if k == 32:
            i = X[x]
            i = i[0:16]
            Xhist = np.zeros([130,2])
            for j in range(len(i)):
                i[j]
                Xhist[j,0] = (np.std(np.histogram(diff(i[j]))[0]))
                Xhist[j,1] = j
            a = Xhist[Xhist[:,0].argsort()]
            b = flip(a)
            b = b[0:16, 0]
            X16_temp = []
            for ii in b: 
                ii = np.array(ii.astype(numpy.int64))
                chan_temp = i[np.array(ii.astype(numpy.int64))]
                chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)
                X16_temp.append(chan_temp[0])           
            X16.append(X16_temp)            
            #X16.append(i[0:16])
            #X16.append(i[16:32])
            i = X[x]
            i = i[16:31]
            Xhist = np.zeros([130,2])
            for j in range(len(i)):
                i[j]
                Xhist[j,0] = (np.std(np.histogram(diff(i[j]))[0]))
                Xhist[j,1] = j
            a = Xhist[Xhist[:,0].argsort()]
            b = flip(a)
            b = b[0:16, 0]
            X16_temp = []
            for ii in b: 
                ii = np.array(ii.astype(numpy.int64))
                chan_temp = i[np.array(ii.astype(numpy.int64))]
                chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)
                X16_temp.append(chan_temp[0])           
            X16.append(X16_temp)            
            y16.append(y[x])
            y16.append(y[x])
            nameKey.append(key[x])
            nameKey.append(key[x])

X = []
for x in X16:
    df = pd.DataFrame(x)
    X.append(df.to_numpy().flatten())
# =============================================================================
# 
# from sklearn.preprocessing import StandardScaler    
# X = pd.DataFrame(X) 
# X = StandardScaler().fit_transform(X)
# y = np.array(y16)
# 
# =============================================================================


#%%
#New control data to be inserted as either a control or as not exhibiting tinnitus

bins = arange(0, 120.1, 0.1) 
ctrl_data_new = numpy.load("ctrl_data_new.npy", allow_pickle=True)
nnames = []
tchannels = []
j = ctrl_data_new[0]['f'] #initializing animal name
animals = []
#bins = arange(0, 120.1, 0.001)
for r in range(len(ctrl_data_new)):
    f = ctrl_data_new[r]['f']
    if f == j:        
        nnames.append(ctrl_data_new[r]['f'])
        tchannels.append(ctrl_data_new[r]['ts'])
    else:
        
        Xhist = np.zeros([130,2])

        for i in range(len(tchannels)):
            Xhist[i,0] = (np.std(np.histogram(diff(tchannels[i]))[0]))
            Xhist[i,1] = i
        a = Xhist[Xhist[:,0].argsort()]
        b = flip(a)
        b = b[0:16, 0]
        X16_temp = []
        for ii in b: 
            ii = np.array(ii.astype(numpy.int64))
            chan_temp = tchannels[np.array(ii.astype(numpy.int64))]
            chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)


            X16_temp.append(chan_temp[0])
        df = pd.DataFrame(X16_temp)
        X.append(df.to_numpy().flatten())
        y16.append(0) #change thia to 2 for multiclass

        animals.append(ctrl_data_new[r-1]['f'])
                
#restart for next time point
        tchannels = []
        tchannels.append(ctrl_data_new[r+1]['ts'])
        nnames.append(ctrl_data_new[r+1]['f'])

    #chan_dict = {k:tchannels}
    j = ctrl_data_new[r]['f']



from sklearn.preprocessing import StandardScaler    
X = pd.DataFrame(X) 
X = StandardScaler().fit_transform(X)
y = np.array(y16)

#%%

#Fitting the model with cross-validation

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
    #plot_confusion_matrix(best_model, X_test, y_test) #plot test results of each part of loop
#summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))    
#%%
plt.figure()
plot_confusion_matrix(best_model, X, y) #plotting all the data
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
ch_list = []
for t in range(len(TestData)):  
    if TestData[t]['ch'] > 31:
        times.append(t)
# =============================================================================
#  #uncomment this to get rid of times under 120
#     elif TestData[t]['t'] < 120: 
#         times.append(t)
# =============================================================================        
    else: 
        time_list.append(TestData[t]['t']) #checking min time (it is 121, which is that we want)
        ch_list.append(TestData[t]['ch']) #checking max channel number (it is 31, which is what we want)
for t in times:
     TestData_copy[t] = 'nan'
 
TestData_new = [x for x in TestData_copy if x != 'nan']     



#%%
from sklearn.preprocessing import StandardScaler 
bins = arange(0, 120.1, 0.1) 
ttimes = []
tchannels = []
j = TestData_new[0]['t']
X16 = []
times = []
animals = []

for r in range(len(TestData_new)):
    k = TestData_new[r]['t']
    if k == j:        
        ttimes.append(TestData_new[r]['t'])
        tchannels.append(TestData_new[r]['ts'])
    else:
# =============================================================================
##code for splitting into two sections of 16
#         tchannels1 = (tchannels[0:16],tchannels[16:32])
#         for jj in (0,1):
#         
#             Xhist = np.zeros([130,2])
#             #chan_dict[k] = tchannels
#             for i in range(len(tchannels1[jj])):
#                 Xhist[i,0] = (np.std(np.histogram(diff(tchannels1[jj][i]))[0]))
#                 Xhist[i,1] = i
#                 a = Xhist[Xhist[:,0].argsort()]
#                 b = flip(a)
#                 b = b[0:16, 0]
#                 X16_temp = []
#             for ii in b: 
#                 ii = np.array(ii.astype(numpy.int64))
#             #print(np.array(ii.astype(numpy.int64)))
#         #print(x[np.array(k.astype(numpy.int64))])
#                 chan_temp = tchannels[np.array(ii.astype(numpy.int64))]
#                 chan_temp = np.histogram(chan_temp[chan_temp<=120],bins)
#             #chan_temp = np.histogram(chan_temp,bins)
# 
#                 X16_temp.append(chan_temp[0])
#             df = pd.DataFrame(X16_temp)
#             X16.append(df.to_numpy().flatten())
#             times.append(TestData_new[r-1]['t'])
#             animals.append(TestData_new[r-1]['f'])
#         
#         
# =============================================================================

# =============================================================================
#code for all 32 channels combined
        Xhist = np.zeros([32,2])
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
            #chan_temp = np.histogram(chan_temp,bins)

            X16_temp.append(chan_temp[0])
        df = pd.DataFrame(X16_temp)
        X16.append(df.to_numpy().flatten())
        times.append(TestData_new[r-1]['t'])
        animals.append(TestData_new[r-1]['f'])       
#end code for all 32 channels combined        
 # ===========================================================================       

        #restart for next time point
        tchannels = []
        tchannels.append(TestData_new[r+1]['ts'])
        ttimes.append(TestData_new[r+1]['t'])

    #chan_dict = {k:tchannels}
    j = TestData_new[r]['t']




#%%
X = pd.DataFrame(X16)
Xtesting1 = StandardScaler().fit_transform(X)     
Xtesting = [] 
for x in Xtesting1:
    Xtesting.append(x)
#create a datarame to remove all nans (shouldn't be applicable anymore but I've kept it just in case)
removeNaNs = pd.DataFrame(
    {'Animal': animals,
     'Times': times,
     'data': Xtesting,
     })

i_timing = []
i_name_key = []
i_newXtest = []

#this loop removes NaNs and creates lists to be used in the next loop
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
#get predictions and probabilities using the best model

testpredictions = best_model.predict(Xtesting)
testprob1 = best_model.predict_proba(Xtesting)
testprob = []
for i in testprob1:
    probsmat = [i[0], i[1]] # for muilticlass add ,i[2]] to probsmat
    testprob.append(probsmat)

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



#Make a dataframe with all of the data together, to use for plotting
New_sorted_data = pd.DataFrame(
    {'Animal': ii_grouped_names,
     'Times': ii_grouped_time,
     'Predictions': ii_grouped_predictions,
     'Probability': ii_grouped_probability,
     })


#%%
#Plotting the figures
import matplotlib.pyplot as plt

for a in range(0, len(New_sorted_data['Animal'])):
    Animal_temp = (New_sorted_data.loc[a])
    datas = numpy.array(Animal_temp['Probability'])
    #figure()
    times = numpy.array(Animal_temp['Times'])

    plt.plot(times,datas[:,1]) #use 'bo' for blue scatterplot
# =============================================================================
#     #uncomment for multiclass
#     plt.plot(times,datas[:,0],'b')
#     plt.plot(times,datas[:,2],'g')    
# =============================================================================
    plt.xlabel('time(s)')
    #plt.xlim(0,max(Animal_temp['Times'])+10)
    plt.ylim(-0.1,1.1)
    plt.ylabel('Probability')
    plt.title("ST Probability of tinnitus for animal " +  str(Animal_temp['Animal'][0]))
    rcParams['pdf.fonttype'] = 42
    savefig('plot_animal' + str(Animal_temp['Animal'][0]) +'.png')

