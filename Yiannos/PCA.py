import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('train_set1.csv')
# df["group"].replace({"NE": "ENT"}, inplace=True)  # mix data-points with no tinnitus to one class
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group', 'avg_ibi', 'avg_spikes_burst',
                              'max_spikes_burst', 'bfr',
                              'p_bursting_spikes', 'p_bursting_time', 'sfr', 'bf', 'sync_n'])  # dropped

features = ['max_sync_bf_dist', 'max_sync_coef', 'mean_sync_bf_dist', 'mean_sync_coef']

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()

X = df_new.loc[:, features].values
X = StandardScaler().fit_transform(X)
y = y[~remove]


pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2',
                                    'principal component 3',
                                    'principal component 4'])

labels = pd.DataFrame(data=y, columns=['group'])
labels["group"].replace({0: "ENT", 1: "ET"}, inplace=True)

finalDf = pd.concat([principalDf, labels], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('4 component PCA, 4 features', fontsize=20)
targets = ['ET', 'ENT']
markers = ['^', 'x']
for target, marker in zip(targets, markers):
    indicesToKeep = finalDf['group'] == target
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1'],
                 finalDf.loc[indicesToKeep, 'principal component 2'],
                 finalDf.loc[indicesToKeep, 'principal component 3'],
                 c=finalDf.loc[indicesToKeep, 'principal component 4'],
                 cmap='BrBG',
                 marker=marker)
                 #s=200*finalDf.loc[indicesToKeep, 'principal component 5'])

ax.legend(targets)
ax.grid()
plt.show()
#plt.savefig('4-component_PCA')

display(pd.DataFrame(pca.components_, columns=df_new.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4']).to_string())
print(pca.explained_variance_ratio_)

