import pickle

import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

'''
df = pd.read_csv('test_set_220727.csv')
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df = df.loc[df["animal"] != 'CW_GP_190826']  # remove data from animal CW_GP_190826 because of weird bf
df_new = df
df_new = df_new.drop(columns=['animal', 't', 'ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n', 'bf_deviation', 'd'])  # dropped  # dropped

#  'd_max', 'bf',   not dropped

df_new = df_new.dropna()
X = StandardScaler().fit_transform(df_new)

sc = SpectralClustering(n_clusters=3, assign_labels='discretize').fit(X)

pred = sc.predict(X)
frame = pd.DataFrame(X)
frame['cluster'] = pred
frame['mark'] = pred
frame["mark"].replace({0: 'r'}, inplace=True)
frame["mark"].replace({1: 'm'}, inplace=True)
frame["mark"].replace({2: 'g'}, inplace=True)

x = frame['d_max']
y = frame['bf']
marks = frame['mark']

for i in range(x.shape[0]):
    plt.scatter(x[i], y[i], color=marks[i], marker='o')

plt.title('Spectral Clustering, 2 Features')
plt.xlabel('d_max')
plt.ylabel('bf')
plt.show()

pickle.dump(gm, open('gaussian_mixture_2ft', 'wb'))
'''

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["group"] != "ET_A"]  # remove ET_T data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET_E": 'ET'}, inplace=True)  # mix data-points with ET

df_new = df
df_new = df_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n'])  # dropped

# 'd_max', 'bf_deviation', 'bf', 'd', group  not dropped

#gm = pickle.load(open('unsupervised/gaussian_mixture_2ft', 'rb'))

df_new = df_new.dropna()
X = StandardScaler().fit_transform(df_new[['d_max', 'bf', 'd', 'bf_deviation']])
pred = GaussianMixture(n_components=3, init_params='random').fit_predict(X)
df_new['cluster'] = pred

print(df_new[['group', 'cluster']].value_counts())

