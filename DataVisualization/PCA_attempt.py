from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pylab import*
import pandas

df=pandas.read_csv('train_set1.csv')



from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
df1=df
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]

y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]

df1=df1.drop(columns=['animal','bf','group'])

remove1=df1.isna().any(axis=1)
df1=df1.dropna()
X = StandardScaler().fit_transform(df1.values)
y=y[~remove1]



# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# =============================================================================

from sklearn.preprocessing import StandardScaler


from sklearn.decomposition import PCA
pca = PCA(.95)
pca.fit(X_train)


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#linear regression using PCA

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(X_train_pca, y_train)
logisticRegr.predict(X_test_pca)
sc = logisticRegr.score(X_test_pca, y_test)
print(f'PCA logistic regression score: {sc}')

logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test)
sc = logisticRegr.score(X_test, y_test)
print(f'logistic regression score: {sc}')

#at least this cell runs *sigh*
# reference: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#%%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

principalDf = pandas.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pandas.concat([principalDf, df[['group']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['ET', 'ENT'] #double checked order
colors = ['g', 'b'] 
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['group'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#see https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60