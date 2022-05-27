import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('train_set1.csv')

df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group'])

# remove = df_new.isna().any(axis=1)
# df_new = df_new.dropna()

features = ['avg_ibi', 'avg_spikes_burst', 'bf', 'bfr', 'max_spikes_burst',
            'max_sync_bf_dist', 'max_sync_coef', 'mean_sync_bf_dist', 'mean_sync_coef',
            'p_bursting_spikes', 'p_bursting_time', 'sfr', 'sync_n']

df_new["avg_ibi"] = df_new["avg_ibi"].fillna(0)
df_new["avg_spikes_burst"] = df_new["avg_spikes_burst"].fillna(0)
df_new["max_spikes_burst"] = df_new["max_spikes_burst"].fillna(0)
df_new["max_sync_bf_dist"] = df_new["max_sync_bf_dist"].fillna(-3000)
df_new["mean_sync_bf_dist"] = df_new["mean_sync_bf_dist"].fillna(-1000)

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()
df = df.dropna()

X = df_new.loc[:, features].values
X = StandardScaler().fit_transform(X)
y = y[~remove]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2'])

finalDf = pd.concat([principalDf, df[['group']]], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['ET', 'ENT', 'NE']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['group'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.savefig('2-component_PCA')

print(pca.explained_variance_ratio_)

