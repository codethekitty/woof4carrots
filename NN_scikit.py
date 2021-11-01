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

y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['animal','bf', 'bfr', 'avg_ibi'])
df1=df1.dropna()
y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['group'])

X = StandardScaler().fit_transform(df1.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#NN model
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

#fit the model
clf = MLPClassifier(random_state=0, max_iter=5000).fit(X_train, y_train)

prob = clf.predict_proba(X_test[:1])
print(prob)

predictions = clf.predict(X_test)
print(predictions)

Sum_wrong_pred =  sum(abs(predictions - y_test))
print("Number of wrong predictions")
print(Sum_wrong_pred)



Percent_wrong = Sum_wrong_pred/size(predictions)
print("Percent of wrong predictions")
print(Percent_wrong)


acc = clf.score(X_test, y_test)
print("mean accuracy")
print(acc)


parameters = clf.get_params()
print("parameters")
print(parameters)


#plot
ys = np.column_stack((y_test,predictions))
plt.figure()
plt.imshow(ys, cmap='hot', interpolation='nearest')
plt.show()

#%%
#attempting different parameters

#fit the model and set parameters
clf = MLPClassifier(random_state=0, max_iter=5000).set_params(epsilon = 0.01, verbose = 'true').fit(X_train, y_train)

#set parameters

prob = clf.predict_proba(X_test[:1])
print(prob)

predictions = clf.predict(X_test)
print(predictions)

Sum_wrong_pred =  sum(abs(predictions - y_test))
print("Number of wrong predictions")
print(Sum_wrong_pred)



Percent_wrong = Sum_wrong_pred/size(predictions)
print("Percent of wrong predictions")
print(Percent_wrong)


acc = clf.score(X_test, y_test)
print("mean accuracy")
print(acc)


parameters = clf.get_params()
print("parameters")
print(parameters)


#plot
ys = np.column_stack((y_test,predictions))
plt.figure()
plt.imshow(ys, cmap='hot', interpolation='nearest')
plt.show()

#%%

# test classification dataset
