import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree

df = pd.read_csv('train_set1.csv')
# df["group"].replace({"NE": "ENT"}, inplace=True)  # mix data-points with no tinnitus to one class
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group', 'avg_ibi', 'avg_spikes_burst',
                              'max_spikes_burst', 'bfr', 'p_bursting_spikes',
                              'p_bursting_time', 'sfr', 'bf', 'sync_n',
                              'max_sync_bf_dist', 'max_sync_coef', 'mean_sync_coef'])  # dropped

# , 'mean_sync_bf_dist' not dropped

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()

X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y[~remove])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Heatmap

'''plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df_new.corr(), dtype=bool))
heatmap = sns.heatmap(df_new.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)

#plt.show()
plt.savefig('triangle_heatmap.png', bbox_inches='tight')'''

# CV
'''
parameters = {'n_estimators': (100, 300, 500), 'max_depth': (5, 10, 40, None),
              'min_samples_split': (2, 3),
              'min_samples_leaf': (1, 2),
              'max_features': (0.5, 0.8, None)}

clf = GridSearchCV(ExtraTreesClassifier(criterion='gini', bootstrap=True, class_weight='balanced_subsample'), parameters, n_jobs=4)
clf.fit(X=X_train, y=np.array(y_train).ravel())
tree_model = clf.best_estimator_
print(clf.best_score_, clf.best_params_)
'''

tree_model = ExtraTreesClassifier(criterion='gini', bootstrap=True, class_weight='balanced_subsample', n_estimators=100)
tree_model.fit(X_train, np.array(y_train).ravel())
print(tree_model.score(X_test, y_test))

'''
filename = 'finalized_model.sav'
pickle.dump(tree_model, open(filename, 'wb'))

fig = plt.figure(figsize=(200, 200))
_ = tree.plot_tree(tree_model.estimators_[0], feature_names=df_new.columns, class_names=['ENT', 'ET'], filled=True)
plt.savefig('tree_1ft_non_standard')

feat_importance = pd.Series(tree_model.feature_importances_, index=df_new.columns)
feat_importance.plot(kind='barh')
plt.savefig('clf_feat_importance', bbox_inches='tight')
'''
