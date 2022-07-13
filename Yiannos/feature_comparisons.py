import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "NE"]  # remove NE data-points
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET_A": "ET"}, inplace=True)  # mix data-points with ET
df["group"].replace({"ET_E": "ET"}, inplace=True)  # mix data-points with ET


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


features = ['animal', 'group', 'loc', 'ch', 'isi_cv', 'sfr',
            'br', 'bdur_max', 'bdur', 'nspikes_burst_max', 'nspikes_burst',
            'p_bursting_time', 'p_bursting_spike', 'ibi_cv',
            'bf_deviation', 'bf', 'sync_n', 'r_max', 'r', 'd_max', 'd']

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'nspikes_burst_max')]
X_max, y_max = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'nspikes_burst')]
X_bdur, y_bdur = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'nspikes_burst', 'nspikes_burst_max')]
X_both, y_both = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'r', 'r_max', 'nspikes_burst_max', 'nspikes_burst')]
X_r, y_r = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation')]
X_base, y_base = make_dataset(ft)

result = [[], [], [], [], []]
for j in range(10):
    res = [[], [], [], [], []]

    for i in range(100):
        res[0].append(run_model(X_max, y_max))
        res[1].append(run_model(X_bdur, y_bdur))
        res[2].append(run_model(X_both, y_both))
        res[3].append(run_model(X_r, y_r))
        res[4].append(run_model(X_base, y_base))

    result[0].append(sum(res[0]) / len(res[0]))
    result[1].append(sum(res[1]) / len(res[1]))
    result[2].append(sum(res[2]) / len(res[2]))
    result[3].append(sum(res[3]) / len(res[3]))
    result[4].append(sum(res[4]) / len(res[4]))


plt.xlabel('Run', fontsize=10)
plt.ylabel('Average Accuracy', fontsize=10)
plt.title('Feature Comparison', fontsize=15)
x = [i for i in range(10)]
plt.plot(x, result[0], label='base + nspikes_burst_max')
plt.plot(x, result[1], label='base + nspikes_burst')
plt.plot(x, result[2], label='base + both')
plt.plot(x, result[3], label='base + both + r')
plt.plot(x, result[4], label='base')
plt.legend(loc='best')

plt.show()
# plt.savefig('clf_ft_comp')
