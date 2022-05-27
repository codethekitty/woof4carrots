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
df_new = df_new.drop(columns=['animal', 'group', 'bf', 'avg_ibi'])  # dropped bf, avg_ibi

features = ['avg_spikes_burst', 'bfr', 'max_spikes_burst',
            'max_sync_bf_dist', 'max_sync_coef', 'mean_sync_bf_dist', 'mean_sync_coef',
            'p_bursting_spikes', 'p_bursting_time', 'sfr', 'sync_n']  # dropped bf, avg_ibi

# df_new["avg_ibi"] = df_new["avg_ibi"].fillna(-5000)
df_new["avg_spikes_burst"] = df_new["avg_spikes_burst"].fillna(-5000)
df_new["max_spikes_burst"] = df_new["max_spikes_burst"].fillna(-5000)
df_new["max_sync_bf_dist"] = df_new["max_sync_bf_dist"].fillna(-100)
df_new["mean_sync_bf_dist"] = df_new["mean_sync_bf_dist"].fillna(-50)

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()
df = df.dropna()

X = df_new.loc[:, features].values
X = StandardScaler().fit_transform(X)
y = y[~remove]


pca = PCA(n_components=5)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2',
                                    'principal component 3',
                                    'principal component 4',
                                    'principal component 5'])

finalDf = pd.concat([principalDf, df[['group']]], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('5 component PCA', fontsize=20)
targets = ['ET', 'ENT']
markers = ['^', 'x']
for target, marker in zip(targets, markers):
    indicesToKeep = finalDf['group'] == target
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1'],
                 finalDf.loc[indicesToKeep, 'principal component 2'],
                 finalDf.loc[indicesToKeep, 'principal component 3'],
                 c=finalDf.loc[indicesToKeep, 'principal component 4'],
                 cmap='BrBG',
                 marker=marker,
                 s=200*finalDf.loc[indicesToKeep, 'principal component 5'])

ax.legend(targets)
ax.grid()
plt.show()
# plt.savefig('5-component_PCA_bf-avg_ibi_dropped')

display(pd.DataFrame(pca.components_, columns=df_new.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']).to_string())
print(pca.explained_variance_ratio_)

