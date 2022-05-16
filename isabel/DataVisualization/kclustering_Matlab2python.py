#MATLAB CODE
# =============================================================================
# clev0 = readtable('cleveland_data_revised-1.xlsx');
# clev = rmmissing(clev0);
# 
# 
# for i = [1:9]
#     idx = kmeans(X,1+i);
#     s = silhouette(X, idx);
#     s_score(i) = mean(s);
# end
# opt_mat = [2:10;s_score]'
# [~,position_clev] = max(s_score)
# optimal_patientgroups = opt_mat(position_clev,1)
# =============================================================================



# =============================================================================
# Python code start
# =============================================================================
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
import numpy as np
from pylab import*
import pandas
k = 3

df=pandas.read_csv('train_set1.csv')
df1 = df
df1 = df1[~df1.animal.str.contains('|'.join(['E5','D4','C8','C2']))]

df1=df1.dropna()
y=unique(df1.loc[:,'group'].values,return_inverse=True)[1]
df1=df1.drop(columns=['animal','bf','group', 'bfr'])
X = StandardScaler().fit_transform(df1.values)

#kmeans with first two columns

km = KMeans(n_clusters=k, random_state=0).fit(X[:,0:1])
l = km.labels_

plt.figure()
km.cluster_centers_


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X[:,0], X[:,1], c=l.astype(float), edgecolor='k')
plt.xlabel('avg ibi')
plt.ylabel('avg spikes burst')
plt.show()

#Kmeans with second two columns
km = KMeans(n_clusters=k, random_state=0).fit(X[:,2:3])
l = km.labels_

km.cluster_centers_

plt.figure()
plt.scatter(X[:,2], X[:,3], c=l.astype(float), edgecolor='k')
plt.xlabel('max spikes burst')
plt.ylabel('max sync bf dist')
plt.show()

#kmeans with 5th and 6th columns
km = KMeans(n_clusters=k, random_state=0).fit(X[:,4:5])
l = km.labels_

km.cluster_centers_


plt.figure()
plt.scatter(X[:,4], X[:,6], c=l.astype(float), edgecolor='k')
plt.xlabel('max sync coef')
plt.ylabel('mean sync bf dist')
plt.show()

#Trying it with y
km = KMeans(n_clusters=k, random_state=0).fit(X[:,0:1])
l = km.labels_

km.cluster_centers_

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1, figsize=(4, 3), auto_add_to_figure=False)

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
fig.add_axes(ax)
ax.scatter(X[:,0], X[:,1], y, c=l.astype(float), edgecolor='k')
ax.set_xlabel('avg ibi')
ax.set_ylabel('avg spikes burst')
ax.set_zlabel('group')


#y vs avg ibi

xy = np.column_stack([X[:,0],y])
km = KMeans(n_clusters=k, random_state=0).fit(xy)
l = km.labels_

km.cluster_centers_

plt.figure()
plt.scatter(xy[:,0], xy[:,1], c=l.astype(float), edgecolor='k')
plt.xlabel('max spikes burst')
plt.ylabel('max sync bf dist')
plt.show()
