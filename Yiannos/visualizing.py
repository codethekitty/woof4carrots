import math
import pickle
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["group"] != "ET_A"]  # remove ET_A data-points
#df = df.loc[df["group"] != "NE"]  # remove NE data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET_E": 'ET'}, inplace=True)  # mix data-points with ET

df_new = df
df_new = df_new.drop(columns=['loc', 'ch', 'isi_cv',  'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'br'])  # dropped

#  'r_max', 'bf', 'd_max', 'd', 'sfr', 'bf_deviation', 'r', 'sync_n' 'group'  not dropped

df_new = df_new.dropna()

df2 = pd.read_csv('mike_data.csv')
df_new2 = df2
df_new2 = df_new2.drop(columns=[ 'ch', 'ind', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'isi_cv', 'br'])  # dropped

#  'r_max', 'd_max', 'bf', 'sync_n', 'd', 'sfr', 'bf_deviation', 'r' 'group' not dropped

df_new2 = df_new2.dropna()

'''
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlabel('d', fontsize=15)
ax.set_ylabel('d_max', fontsize=15)
ax.set_title('Train Data vs Test Distribution', fontsize=20)
targets = ['ET', 'ENT', 'NE']
colors = ['r', 'b', 'g']
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
'''
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('mean d_max', fontsize=15)
ax.set_ylabel('mean d', fontsize=15)
ax.set_zlabel('mean sync_n', fontsize=15)
ax.set_title('Data Distribution by Animal', fontsize=20)

animals1 = df_new['animal'].unique()
animals2 = df_new2['animal'].unique()

et_points = []
ent_points = []
ne_points = []

for animal in animals1:
    df_a = df_new.loc[df_new["animal"] == animal]
    color = 'r'
    status = df_a['group'].values[0]
    if status == 'ENT':
        color = 'b'
    elif status == 'NE':
        color = 'g'
    marker = 'x'
    d_avg = df_a['d_max'].mean()
    dmax_avg = df_a['d'].mean()
    bf_avg = df_a['sync_n'].mean()
    if status == 'ET':
        et_points.append((d_avg, dmax_avg, color, marker, bf_avg))
    elif status == 'ENT':
        ent_points.append((d_avg, dmax_avg, color, marker, bf_avg))
    else:
        ne_points.append((d_avg, dmax_avg, color, marker, bf_avg))


for animal in animals2:
    df_a = df_new2.loc[df_new2["animal"] == animal]
    color = 'r'
    status = df_a['group'].values[0]
    if status == 'ENT':
        color = 'b'
    elif status == 'NE':
        color = 'g'
    marker = 'o'
    d_avg = df_a['d_max'].mean()
    dmax_avg = df_a['d'].mean()
    bf_avg = df_a['sync_n'].mean()
    if status == 'ET':
        et_points.append((d_avg, dmax_avg, color, marker, bf_avg))
    elif status == 'ENT':
        ent_points.append((d_avg, dmax_avg, color, marker, bf_avg))
    else:
        ne_points.append((d_avg, dmax_avg, color, marker, bf_avg))

xs1 = [x[0] for x in et_points]
ys1 = [y[1] for y in et_points]
zs1 = [z[4] for z in et_points]
colors1 = [c[2] for c in et_points]
markers1 = [m[3] for m in et_points]

xs2 = [x[0] for x in ent_points]
ys2 = [y[1] for y in ent_points]
zs2 = [z[4] for z in ent_points]
colors2 = [c[2] for c in ent_points]
markers2 = [m[3] for m in ent_points]

xs3 = [x[0] for x in ne_points]
ys3 = [y[1] for y in ne_points]
zs3 = [z[4] for z in ne_points]
colors3 = [c[2] for c in ne_points]
markers3 = [m[3] for m in ne_points]

ax.scatter3D(xs1, ys1, zs1, c=colors1)
ax.scatter3D(xs2, ys2, zs2,  c=colors2)
ax.scatter3D(xs3, ys3, zs3,  c=colors3)

ax.legend(['ET', 'ENT', 'NE'])
ax.grid()
plt.show()


