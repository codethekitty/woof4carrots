import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('test_set_220727.csv')
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df = df.loc[df["animal"] != 'CW_GP_190826']  # remove data from animal CW_GP_190826 because of weird bf
df_new = df
df_new = df_new.drop(columns=['animal', 't', 'ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n'])  # dropped  # dropped

# , 'bf_deviation', 'bf', 'd_max', 'd'  not dropped

df_new = df_new.dropna()
X = StandardScaler().fit_transform(df_new)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

pred = kmeans.predict(X)
frame = pd.DataFrame(X)
frame['cluster'] = pred
frame.columns = ['d_max', 'bf', 'd', 'bf_deviation', 'cluster']

'''
color = ['blue', 'green', 'cyan']
for k in range(0, 3):
    data = frame[frame["cluster"] == k]
    plt.scatter(data["d_max"], data["bf"], c=color[k])
plt.show()
'''


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('d_max', fontsize=15)
ax.set_ylabel('bf', fontsize=15)
ax.set_zlabel('d', fontsize=15)
ax.set_title('Kmeans Clustering, 4 features', fontsize=20)
targets = [0, 1, 2]
markers = ['^', 'x', 'o']
for target, marker in zip(targets, markers):
    indicesToKeep = frame['cluster'] == target
    ax.scatter3D(frame.loc[indicesToKeep, 'd_max'],
                 frame.loc[indicesToKeep, 'bf'],
                 frame.loc[indicesToKeep, 'd'],
                 c=frame.loc[indicesToKeep, 'bf_deviation'],
                 cmap='BrBG',
                 marker=marker)
                 #s=200*finalDf.loc[indicesToKeep, 'principal component 5'])

ax.legend(targets)
ax.grid()
plt.show()
