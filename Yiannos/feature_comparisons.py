import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_set1.csv')
df = df.loc[df["group"] != "NE"]  # remove NE data-points


def make_dataset(drop_features):
    df_new = df
    y = np.unique(df_new.loc[:, 'group'].values, return_inverse=True)[1]
    df_new = df_new.drop(columns=drop_features)  # dropped

    remove = df_new.isna().any(axis=1)
    df_new = df_new.dropna()

    X = StandardScaler().fit_transform(df_new)
    y = pd.DataFrame(y[~remove])

    return X, y


def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    tree_model = ExtraTreesClassifier(criterion='gini', bootstrap=True,
                                      class_weight='balanced_subsample',
                                      n_estimators=100)
    tree_model.fit(X_train, np.array(y_train).ravel())

    return tree_model.score(X_test, y_test)


features = ['animal', 'group',
            'avg_ibi', 'avg_spikes_burst', 'max_spikes_burst',
            'bfr', 'p_bursting_spikes', 'p_bursting_time',
            'sfr', 'bf', 'sync_n', 'max_sync_bf_dist',
            'max_sync_coef', 'mean_sync_bf_dist', 'mean_sync_coef']

X_all, y_all = make_dataset(['animal', 'group'])

ft = features
ft.remove('mean_sync_bf_dist')
X_mean_sync_dist, y_mean_sync_dist = make_dataset(ft)

ft = features
ft.remove('mean_sync_coef')
X_mean_sync_coef, y_mean_sync_coef = make_dataset(ft)

ft = [feat for feat in features if feat not in ('max_sync_bf_dist', 'max_sync_coef',
                                                'mean_sync_bf_dist', 'mean_sync_coef')]
X_4feat, y_4feat = make_dataset(ft)

res = [[], [], [], []]

for i in range(500):
    res[0].append(run_model(X_all, y_all))
    res[1].append(run_model(X_mean_sync_dist, y_mean_sync_dist))
    res[2].append(run_model(X_mean_sync_coef, y_mean_sync_coef))
    res[3].append(run_model(X_4feat, y_4feat))

plt.xlabel('Run', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Feature Comparison', fontsize=15)
x = [i for i in range(500)]
plt.plot(x, res[0], label='all features')
plt.plot(x, res[1], label='mean_sync_bf_dist')
plt.plot(x, res[2], label='mean_sync_coef')
plt.plot(x, res[3], label='mean+max bf_dist+coef')
plt.legend(loc='best')

plt.show()
# plt.savefig('clf_feature_comparisons')
