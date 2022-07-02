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

ft = [feat for feat in features if feat not in ('max_sync_bf_dist', 'max_sync_coef',
                                                'mean_sync_bf_dist', 'mean_sync_coef', 'bf', 'sync_n', 'sfr')]
X_sfr, y_sfr = make_dataset(ft)

ft = [feat for feat in features if feat not in ('max_sync_bf_dist', 'max_sync_coef',
                                                'mean_sync_bf_dist', 'mean_sync_coef', 'bf', 'sync_n', 'p_bursting_time')]
X_burst_time, y_burst_time = make_dataset(ft)

ft = [feat for feat in features if feat not in ('max_sync_bf_dist', 'max_sync_coef',
                                                'mean_sync_bf_dist', 'mean_sync_coef', 'bf', 'sync_n', 'sfr', 'p_bursting_time')]
X_both, y_both = make_dataset(ft)

ft = [feat for feat in features if feat not in ('max_sync_bf_dist', 'max_sync_coef',
                                                'mean_sync_bf_dist', 'mean_sync_coef', 'bf', 'sync_n')]
X_base, y_base = make_dataset(ft)

result = [[], [], [], []]
for j in range(10):
    res = [[], [], [], []]

    for i in range(100):
        res[0].append(run_model(X_sfr, y_sfr))
        res[1].append(run_model(X_burst_time, y_burst_time))
        res[2].append(run_model(X_both, y_both))
        res[3].append(run_model(X_base, y_base))

    result[0].append(sum(res[0]) / len(res[0]))
    result[1].append(sum(res[1]) / len(res[1]))
    result[2].append(sum(res[2]) / len(res[2]))
    result[3].append(sum(res[3]) / len(res[3]))


plt.xlabel('Run', fontsize=10)
plt.ylabel('Average Accuracy', fontsize=10)
plt.title('Feature Comparison', fontsize=15)
x = [i for i in range(10)]
plt.plot(x, result[0], label='mean+max+sync+bf + sfr')
plt.plot(x, result[1], label='mean+max+sync+bf + p_burst_time')
plt.plot(x, result[2], label='mean+max+sync+bf + both')
plt.plot(x, result[3], label='baseline')
plt.legend(loc='best')

plt.show()
# plt.savefig('clf_ft_comp')
