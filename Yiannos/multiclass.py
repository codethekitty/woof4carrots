import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('train_set.csv')
df = df.loc[df["group"] != "ET_T"]  # remove ET_T data-points
df = df.loc[df["ch"] <= 31]  # remove data from channels past 31
df["group"].replace({"ET": 2}, inplace=True)
df["group"].replace({"ET_A": 2}, inplace=True)  # mix data-points with ET
df["group"].replace({"ET_E": 2}, inplace=True)  # mix data-points with ET
df["group"].replace({"ENT": 1}, inplace=True)
df["group"].replace({"NE": 0}, inplace=True)

df_new = df
df_new = df_new.drop(columns=['animal', 'loc', 'ch', 'isi_cv', 'sfr', 'br', 'bdur_max', 'bdur',
                              'nspikes_burst_max', 'nspikes_burst', 'p_bursting_time', 'p_bursting_spike',
                              'ibi_cv', 'r_max', 'r', 'sync_n'])  # dropped

# , 'bf_deviation', 'bf', 'd_max', 'd'  not dropped

df_new = df_new.dropna()
y = df_new.loc[:, "group"]
df_new = df_new.drop(columns=['group'])
X = StandardScaler().fit_transform(df_new)
y = pd.DataFrame(y)

tree_ovr = OneVsOneClassifier(ExtraTreesClassifier(bootstrap=True, class_weight='balanced_subsample'))
knn_ovr = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1, weights='distance'))
svm_ovr = OneVsOneClassifier(SVC(class_weight='balanced', C=10000))

# Confusion Matrix
'''
disp = ConfusionMatrixDisplay.from_estimator(
        tree_ovr,
        X_test,
        y_test,
        display_labels=['NE', 'ENT', 'ET'],
        cmap=plt.cm.Blues,
    )

print(disp.confusion_matrix)
plt.show()
'''

# Save Model
'''
pickle.dump(tree_ovr, open('tree_ovo_rmax', 'wb'))
pickle.dump(knn_ovr, open('knn_ovo_rmax', 'wb'))
pickle.dump(svm_ovr, open('svm_ovo_rmax', 'wb'))
'''

# Model Comparison
'''
result = [[], [], [], []]
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree_ovr.fit(X_train, np.array(y_train).ravel())
    svm_ovr.fit(X_train, np.array(y_train).ravel())
    knn_ovr.fit(X_train, np.array(y_train).ravel())

    result[0].append(tree_ovr.score(X_test, y_test))
    result[1].append(svm_ovr.score(X_test, y_test))
    result[2].append(knn_ovr.score(X_test, y_test))

plt.xlabel('Run', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Model Comparison', fontsize=15)
x = [i for i in range(100)]
plt.plot(x, result[0], label='Tree_ovr')
plt.plot(x, result[1], label='SVM_ovr')
plt.plot(x, result[2], label='KNN_ovr')
plt.legend(loc='best')

plt.show()
'''

# Feature comparison
'''
def make_dataset(drop_features):
    df_new = df
    df_new = df_new.drop(columns=drop_features)  # dropped
    df_new = df_new.dropna()
    y = df_new.loc[:, "group"]
    df_new = df_new.drop(columns=['group'])
    X = StandardScaler().fit_transform(df_new)
    y = pd.DataFrame(y)
    return X, y


def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    tree_ovr = OneVsOneClassifier(ExtraTreesClassifier(bootstrap=True, class_weight='balanced_subsample'))
    knn_ovr = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1, weights='distance'))
    svm_ovr = OneVsOneClassifier(SVC(class_weight='balanced', C=10000))

    tree_ovr.fit(X_train, np.array(y_train).ravel())

    return tree_ovr.score(X_test, y_test)


features = ['animal', 'loc', 'ch', 'isi_cv', 'sfr',
            'br', 'bdur_max', 'bdur', 'nspikes_burst_max', 'nspikes_burst',
            'p_bursting_time', 'p_bursting_spike', 'ibi_cv',
            'bf_deviation', 'bf', 'sync_n', 'r_max', 'r', 'd_max', 'd']

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'r', 'r_max')]
X_max, y_max = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'r_max')]
X_bdur, y_bdur = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation', 'r')]
X_r, y_r = make_dataset(ft)

ft = [feat for feat in features if feat not in ('d', 'd_max', 'bf', 'bf_deviation')]
X_base, y_base = make_dataset(ft)

result = [[], [], [], [], []]
for j in range(20):
    res = [[], [], [], [], []]

    for i in range(100):
        res[0].append(run_model(X_max, y_max))
        res[1].append(run_model(X_bdur, y_bdur))
        res[2].append(run_model(X_r, y_r))
        res[3].append(run_model(X_base, y_base))

    result[0].append(sum(res[0]) / len(res[0]))
    result[1].append(sum(res[1]) / len(res[1]))
    result[2].append(sum(res[2]) / len(res[2]))
    result[3].append(sum(res[3]) / len(res[3]))


plt.xlabel('Run', fontsize=10)
plt.ylabel('Average Accuracy', fontsize=10)
plt.title('Feature Comparison', fontsize=15)
x = [i for i in range(20)]
plt.plot(x, result[0], label='base + both')
plt.plot(x, result[1], label='base + r_max')
plt.plot(x, result[2], label='base + r')
plt.plot(x, result[3], label='base')
plt.legend(loc='best')

plt.show()
'''

