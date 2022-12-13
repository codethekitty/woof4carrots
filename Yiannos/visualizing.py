import pickle
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["group"] != "ET_A"]  # remove ET_A data-points
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET_E": 'ET'}, inplace=True)  # mix data-points with ET

df_new = df
df_new = df_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv',  'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'br'])  # dropped

#  'r_max', 'bf', 'd_max', 'd', 'sfr', 'bf_deviation', 'r', 'sync_n' 'group'  not dropped

df_new = df_new.dropna()

df2 = pd.read_csv('mike_data.csv')
df_new2 = df2
df_new2 = df_new2.drop(columns=['animal', 'ch', 'ind', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'isi_cv', 'br'])  # dropped

#  'r_max', 'd_max', 'bf', 'sync_n', 'd', 'sfr', 'bf_deviation', 'r' 'group' not dropped

df_new2 = df_new2.dropna()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('d', fontsize=15)
ax.set_ylabel('d_max', fontsize=15)
ax.set_title('Train Data vs Test Distribution', fontsize=20)
targets = ['ET', 'ENT']
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = df_new['group'] == target
    ax.scatter(df_new.loc[indicesToKeep, 'd'],
                 df_new.loc[indicesToKeep, 'd_max'],
                 c=color,
                 marker='x')
for target, color in zip(targets, colors):
    indicesToKeep = df_new2['group'] == target
    ax.scatter(df_new2.loc[indicesToKeep, 'd'],
                 df_new2.loc[indicesToKeep, 'd_max'],
                 c=color,
                 marker='o')

ax.legend(targets)
ax.grid()
plt.show()



