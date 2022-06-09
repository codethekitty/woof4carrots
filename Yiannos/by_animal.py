import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train_set1.csv')
df = df.loc[df["group"] != "NE"]  # remove NE data-points

df_new = df
df_new = df_new.drop(columns=['group'])  # dropped

features = ['avg_ibi', 'avg_spikes_burst', 'bf', 'bfr', 'max_spikes_burst',
            'max_sync_bf_dist', 'max_sync_coef', 'mean_sync_bf_dist', 'mean_sync_coef',
            'p_bursting_spikes', 'p_bursting_time', 'sfr', 'sync_n']

df_new["avg_ibi"] = df_new["avg_ibi"].fillna(-5000)
df_new["avg_spikes_burst"] = df_new["avg_spikes_burst"].fillna(-5000)
df_new["max_spikes_burst"] = df_new["max_spikes_burst"].fillna(-5000)
df_new["max_sync_bf_dist"] = df_new["max_sync_bf_dist"].fillna(-100)
df_new["mean_sync_bf_dist"] = df_new["mean_sync_bf_dist"].fillna(-50)

df_new = df_new.dropna()


def extract_animal(dff, animal, feat):
    ex_an = dff.loc[dff['animal'] == animal]
    return ex_an.loc[:, feat].values.ravel()


animals = ['B1', 'B6', 'B9', 'C2', 'C4', 'C5', 'C7', 'C8', 'D10', 'D3', 'D4', 'D5', 'D8', 'E2', 'E4', 'E5', 'E6', 'E7',
           'E8']

X = []
for anim in animals:
    X.append(extract_animal(df_new, anim, features)[:273])
y = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
X = pd.DataFrame(data=X)
X = StandardScaler().fit_transform(X)

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
ax.set_title('4 component PCA by Animal', fontsize=20)
targets = ['ET', 'ENT']
markers = ['^', 'x']
for target, marker in zip(targets, markers):
    indicesToKeep = finalDf['group'] == target
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1'],
                 finalDf.loc[indicesToKeep, 'principal component 2'],
                 finalDf.loc[indicesToKeep, 'principal component 4'],
                 c=finalDf.loc[indicesToKeep, 'principal component 3'],
                 cmap='spring',
                 marker=marker)

ax.legend(targets)
ax.grid()
# plt.show()
# plt.savefig('4-component_PCA_by_animal')

# display(pd.DataFrame(pca.components_, index=['PC-1', 'PC-2', 'PC-3', 'PC-4']).to_string())
# print(pca.explained_variance_ratio_)

'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Extra Trees
parameters = {}
clf = GridSearchCV(ExtraTreesClassifier(criterion='gini', max_depth=70, bootstrap=True, class_weight='balanced_subsample', n_estimators=100, min_samples_split=10), parameters, n_jobs=4)
clf.fit(X=X_train, y=np.array(y_train).ravel())
tree_model = clf.best_estimator_
print(clf.best_score_, clf.best_params_)

print(tree_model.score(X_test, y_test))'''


