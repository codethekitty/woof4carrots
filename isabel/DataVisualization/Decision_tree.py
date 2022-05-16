#Decision tree attempt
#scikit documentation referenced
from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas
from sklearn.model_selection import train_test_split
from sklearn import tree

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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#create and fit tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

#plot tree 
from matplotlib import pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(clf)
#https://stackoverflow.com/questions/59174665/how-can-i-adjust-the-size-of-the-plot-tree-graph-in-sklearn-to-make-it-readable

#test tree on new parameters
y_pred = clf.predict(X_test)

from scipy.stats import pearsonr
corr, _ = pearsonr(y_test, y_pred)

print(f"Decision Tree correlation: {corr}")


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Decision Tree mean squared error: {mse}')


import matplotlib.pyplot as plt


#Graph of test y values and predicted y values
ys = np.column_stack((y_test,y_pred))
plt.figure()
plt.imshow(ys, cmap='hot', interpolation='nearest')
plt.show()
#%%
## Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)

y_predRF = rf.predict(X_test)

corr, _ = pearsonr(y_test, y_predRF)
print(f'Random Forest correlation: {corr}')


mse = mean_squared_error(y_test, y_predRF)
print(f'Random Forest mean squared error: {mse}')




#change graphs from new figure to in the console
#%matplotlib auto
#%matplotlib inline
