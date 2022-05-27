import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('train_set1.csv')
# df["group"].replace({"NE": "ENT"}, inplace=True)  # mix data-points with no tinnitus to one class
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df_new = df
y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
df_new = df_new.drop(columns=['animal', 'group', 'bf', 'avg_ibi'])  # dropped bf, avg_ibi

# df_new["avg_ibi"] = df_new["avg_ibi"].fillna(-5000)
df_new["avg_spikes_burst"] = df_new["avg_spikes_burst"].fillna(-5000)
df_new["max_spikes_burst"] = df_new["max_spikes_burst"].fillna(-5000)
df_new["max_sync_bf_dist"] = df_new["max_sync_bf_dist"].fillna(-100)
df_new["mean_sync_bf_dist"] = df_new["mean_sync_bf_dist"].fillna(-50)

remove = df_new.isna().any(axis=1)
df_new = df_new.dropna()
df = df.dropna()

X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y[~remove])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df_new.corr(), dtype=bool))
heatmap = sns.heatmap(df_new.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)

plt.show()
# plt.savefig('triangle_heatmap.png', bbox_inches='tight')'''


# Extra Trees
model = ExtraTreesClassifier(criterion='gini', n_estimators=300, bootstrap=True, class_weight='balanced_subsample')
model.fit(X_train, np.array(y_train).ravel())

print(model.score(X_test, y_test))

fig = plt.figure(figsize=(200, 200))
_ = tree.plot_tree(model.estimators_[0], feature_names=df_new.columns, class_names=['ENT', 'ET'], filled=True)
plt.savefig('decision_tree')

'''feat_importance = pd.Series(model.feature_importances_, index=df_new.columns)
feat_importance.plot(kind='barh')
plt.savefig('extratrees_clf_feat_import', bbox_inches='tight')'''
